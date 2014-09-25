################################################################################
# Support Vector Regression using Evolutionary Strategy                        #
# SVR-ES                                                                       #
################################################################################
#                                                                              #
# Details:                                                                     #
#                                                                              #
#     Name:      SVR-ES.R                                                      #
#     Type:      R source code                                                 #
#     Version:   0.2                                                           #
#     Date:      2012-02-10 (modifications by AJC 2012-10-02)                  #
#     Dependencies: qualV, e1071, FNN, Boruta, CaDENCE                         #
#                                                                              #
# Author(s):                                                                   #
#                                                                              #
#     Aranildo R. Lima  <arodrigu@eos.ubc.ca>                                  #
#     Alex J. Cannon                                                           #
#                                                                              #
# References:                                                                  #
#     Aranildo R. Lima, Alex J. Cannon, William W. Hsieh, Nonlinear regression #
#     in environmental sciences by support vector machines combined with       #
#     evolutionary strategy, Computers & Geosciences, Volume 50, January 2013, #
#     http://www.sciencedirect.com/science/article/pii/S0098300412002269)      #
#                                                                              #
#     Cherkassky, V. and Ma, Y., 2004. Practical selection of SVM parameters   #
#     and noise estimation for SVM regression. Neural Netw. 17, 113–126.       #
#                                                                              #
#     Eiben, A.E. and Smith, J.E.,2003. Introduction to Evolutionary           #
#     Computing. Natural Computing Series, Springer, Berlin.                   #
#                                                                              #
#                                                                              #
################################################################################

library("FNN")
require("qualV")
library("e1071")
require("Boruta")
require("CaDENCE")

train.svr <- 
  #separates crossvalidation and split 
  function(xtrain, ytrain, hypparameter, ErrorFunc, PercentValid=20, kfold=1,
           SplitRandom=FALSE, kernel="radial"){
  
  if(kfold>1){
    n.cases =nrow(xtrain)
    index.block <- xval.buffer(n.cases, kfold)
    pred.valid <- rep(Inf, n.cases)

    for(nb in 1:kfold){
        svr.try <- try(svm(xtrain[index.block[[nb]]$train,,drop=FALSE], 
                           ytrain[index.block[[nb]]$train,,drop=FALSE],
                            kernel=kernel, gamma=hypparameter[1], 
                            epsilon=hypparameter[2], cost=hypparameter[3]),
                       silent=TRUE)
        
        if(class(svr.try)!='try-error'){
          pred.valid[index.block[[nb]]$valid] = predict(svr.try, xtrain[index.block[[nb]]$valid,,drop=FALSE])
        }else{
          return (list(error.svm=TRUE))
        }
    }#end for crossvalidation  
    return (list(error.svm=FALSE, fitted=pred.valid,ffValid= ErrorFunc(ytrain, pred.valid)))
  }else{
    nTrain = nrow(xtrain)
    indValid <- nTrain-round((nTrain*(PercentValid/100)))
    if(SplitRandom){
      cases <- sample(nTrain)
      x.fit <- xtrain[cases,,drop=FALSE]
      y.fit <- ytrain[cases]
    }
    x.fit.train <- xtrain[1:indValid,,drop=FALSE]
    x.fit.valid <- xtrain[(indValid+1):nTrain,,drop=FALSE]
    y.fit.train <- ytrain[1:indValid]
    y.fit.valid <- ytrain[(indValid+1):nTrain]
    
    svr.try <- try(svm(x.fit.train,y.fit.train,kernel="radial",
              gamma=hypparameter[1], epsilon=hypparameter[2],
              cost=hypparameter[3]),
              silent=TRUE)

    if(class(svr.try)!='try-error'){
      sv <- svr.try
      return (list(error.svm=FALSE, fitted.train=sv$fitted, fitted.valid=predict(sv, x.fit.valid), 
                  ffValid= ErrorFunc(y.fit.valid, predict(sv, x.fit.valid)),
                  ffTrain = ErrorFunc(y.fit.train, sv$fitted)))
    }else{
      return (list(error.svm=TRUE))
    }
  }#end if crossvalidation 
}#end function train.svr

svres <-
  # Hybrid method combining Support Vector Regression and Evolutionary Strategy
  function(X.train, Y.train, X.test, PercentValid=20, Generations=500,
           InitialGamma=0.001, ErrorFunc=MSE, Step=FALSE, StepBoruta=FALSE,
           SplitRandom=FALSE, Trace=TRUE, dTrace=10,earlyStop=50,kfold=1)
  {
    if(Step & !StepBoruta) {
      #################### prescreening predictors ###########################
      targetTrain <- as.matrix(Y.train)
      colnames(targetTrain) <- c("Y")
      Xtarget <- cbind(X.train, targetTrain)
      fit.step <- lm(Y ~ ., as.data.frame(Xtarget))
      para.step <- step(fit.step, trace = 0)
      step.coef <- attr(para.step$terms, 'term.labels')
      if(length(step.coef)==0){
        step.coef <- colnames(Xtarget)[which.max(abs(fit.step$coef[-1]))]
      }
      x.fit <- X.train[,step.coef,drop=FALSE]
      x.test <- X.test[,step.coef,drop=FALSE]
      stepvars <- colnames(X.train) %in% step.coef
    }else if(Step & StepBoruta){
      trf <- suppressWarnings(tuneRF(X.train, Y.train, doBest=TRUE,
                                     plot=FALSE, ntreeTry=500, nodesize=1))
      bor <- Boruta(x=as.data.frame(X.train), y=Y.train, light=FALSE,
                    doTrace=2, ntree=trf$ntree, mtry=trf$mtry, nodesize=1)
      bor <- suppressWarnings(TentativeRoughFix(bor))
      stepvars <- abs(as.integer(bor$finalDecision)-3)
      stepvars <- stepvars==1
      x.fit <- X.train[,stepvars,drop=FALSE]
      x.test <- X.test[,stepvars,drop=FALSE]
    }else{
      x.fit <- X.train
      x.test <- X.test
      stepvars <- rep(TRUE, ncol(X.train))
    }
    y.fit <- Y.train
    nTrain <- length(y.fit)
    ######### initializing with Cherkassky-Ma parameters #######################
    nNeighbor <- 4
    kksvr <- knn.reg(x.fit,test=NULL,y=y.fit,k=nNeighbor)
    factor1 <- ((nTrain)^(1/5))*nNeighbor
    factor2 <- ((nTrain)^(1/5))*(nNeighbor-1)
    errorEst <- sqrt((factor1/factor2)*(kksvr$PRESS/nTrain))
    epsilCal <- 3*errorEst*(sqrt(log(nTrain)/nTrain))
    C <- max(abs(mean(y.fit) + 3*sd(c(y.fit))),abs(mean(y.fit) - 3*sd(c(y.fit))))
    
    ##########  Initializing FF ################################################
    hypparameter <- c(InitialGamma,epsilCal, C)
    names(hypparameter) <- c("Gamma", "Epsilon", "C")
    
    fitted.svr <- train.svr(x.fit, y.fit, hypparameter, ErrorFunc, PercentValid, kfold)
    ffValid <- fitted.svr$ffValid
    
    if(kfold == 1){
      ffTrain <- fitted.svr$ffTrain
    }
    mutation <- hypparameter
    noevolution <- 0
    ############ running the generations #######################################
    for (d in 1:Generations) {
      noevolution = noevolution +1
      firstNorm <- rnorm(3)
      son <- abs(hypparameter+(mutation*firstNorm))
      tau <- 1/sqrt(2*sqrt(3))
      taul <- 1/sqrt(6)
      firstNorm <- rnorm(1)
      secNorm <- rnorm(3)
      mutationtemp <- mutation*exp(taul*firstNorm+tau*secNorm)
      fitted.svr <- train.svr(x.fit, y.fit, son, ErrorFunc, PercentValid, kfold)
      
      if(fitted.svr$error.svm){
        ffSonValid <- Inf
      } else{
        ffSonValid <- fitted.svr$ffValid
      }
      
      if(ffSonValid < ffValid){
        if(kfold > 1 ){
          hypparameter <- son
          mutation <- mutationtemp
          ffValid <- ffSonValid
          noevolution <- 0
        }else{
          ffSonTrain <- fitted.svr$ffTrain
          if (ffSonTrain < ffTrain){
            hypparameter <- son
            mutation <- mutationtemp
            ffValid <- ffSonValid
            ffTrain <- ffSonTrain
            noevolution <- 0
          }#end if train
        }#end if flod (crossvalidation)
      }#end if valid
      
      if(Trace & ((d%%dTrace)==0)){
        cat('Generation --> ', d, "\n")
        if(kfold > 1 ){
          cat("Error: Valid --> ", ffValid,"\n")
        }else{
          cat("Error: Train -->",  ffTrain, " | Valid --> ", ffValid,"\n")
        }
      }
      if(noevolution > earlyStop){
        break
      }
    } # end for generations
    ############## calculating the targets #####################################
    sv <- svm(x.fit,y.fit,kernel="radial", gamma=hypparameter[1],
              epsilon=hypparameter[2], cost=hypparameter[3])
    p.svmES <- predict(sv, x.test)
    
    if(kfold > 1 ){
      svres.results <- list(hypparameter=hypparameter, forecast=p.svmES,
                            svmf=sv, ffTrain=NULL, ffValid=ffValid,
                            stepvars=stepvars)
    }else{
      svres.results <- list(hypparameter=hypparameter, forecast=p.svmES,
                            svmf=sv, ffTrain=ffTrain, ffValid=ffValid,
                            stepvars=stepvars)
    }
    return (svres.results)
  }

################################################################################    

#rm(list=ls(all=TRUE))

#################### loading and adjusting the dataset #########################
# data(FraserSediment, package='CaDENCE')
# x.train <- FraserSediment$x.1970.1976
# y.train <- log10(FraserSediment$y.1970.1976)
# x.test <- FraserSediment$x.1977.1979
# y.test <- log10(FraserSediment$y.1977.1979)
# 
# x.mn <- colMeans(x.train)
# x.sd <- apply(x.train, 2, sd)
# y.max <- max(y.train)
# 
# x.trainsd <- sweep(sweep(x.train, 2, x.mn, '-'), 2, x.sd, '/')
# x.testsd <- sweep(sweep(x.test, 2, x.mn, '-'), 2, x.sd, '/')
# 
# y.trainsd <- y.train/y.max
# y.testsd <- y.test/y.max
# 
# ######################### using only 3000 points #################################
# resul <- svres(x.trainsd, y.trainsd, x.testsd, Step=FALSE,kfold=5)
# MSE(y.testsd,resul$forecast)
# MAE(y.testsd,resul$forecast)
# 
# plot(y.testsd[1:100])
# lines(resul$forecast[1:100])

################################################################################
##  ELM (Extreme Learning Machines)                                         ####
##  with ensemble, validation and penalty parameter                         ####
##                                                                          ####
# Details:                                                                     #
#                                                                              #
#     Name:      elm.R                                                         #
#     Type:      R source code                                                 #
#     Version:   0.1                                                           #
#     Date:      2014-06-20                                                    #
#     Dependencies: qualV, MASS, CaDENCE                                       #
#                                                                              #
# Author(s):                                                                   #
#                                                                              #
#     Aranildo R. Lima  <arodrigu@eos.ubc.ca>                                  #
#                                                                              #
# References:                                                                  #
#       G.-B. Huang, Q.-Y. Zhu, C.-K. Siew (2006)                              #
#       Extreme learning machine: Theory and applications                      #
#       Neurocomputing 70 (2006) 489-501                                       #
#                                                                              #
#                                                                              #
################################################################################

library(qualV)
library(MASS)
library(CaDENCE)
#################################################################################################
#                                                                                               #
#                                       ELM Starts Here                                         #
#                                                                                               #
#################################################################################################
Elm.optmization <- function(X.fit, Y.fit, Number.hn=10, autorangeweight=FALSE, rangeweight=NULL, PenaltyP=0, activation='TANH',outputBias = FALSE){
  ############## carregando os valores ###################
  X.fit <- t(X.fit)
  Target <- Y.fit
  n.InputNeurons=nrow(X.fit)
  n.TrainingData=ncol(X.fit)
  ############################### Random generate input weights inputWeight (w_i) and biases biasofHN (b_i) of hidden neurons
  if(autorangeweight){
    if (is.null(rangeweight)){
      switch(activation,
             'TANH' = {a=1},
                      {a=2}
              )
      rangew <- a*(n.InputNeurons)^-.5 # automatic range of the weights based on the number of predictors  
    }else{
      rangew <- rangeweight
    }  
  }else{
    rangew <- 1
    a <- 1
  }
  inputWeight=matrix(runif(Number.hn*n.InputNeurons, min=-1, max=1),Number.hn, n.InputNeurons)*rangew
  biasofHN=runif(Number.hn, min=-1, max=1)*a
  
  tempH=inputWeight%*%X.fit
  ind=matrix(1,1,n.TrainingData)
  BiasMatrix=biasofHN%*%ind      # Extend the bias matrix biasofHN to match the demention of H
  #clean a memory
  rm(ind) 
  tempH=tempH+BiasMatrix
  rm(BiasMatrix)
  
  ############# Train ######################
  switch(activation,
         'TANH' = {H=t(tanh(tempH))},
         {H= t(1 / (1 + exp(-tempH)))}
         )
  #clean a memory
  rm(tempH)

  if(outputBias){
    H <- cbind(1,H)    
  }

  if (PenaltyP==0)
  {
    outputWeight=(ginv(H) %*% Target)         # implementation without regularization factor //refer to 2006 Neurocomputing paper
  }else{
    #outputWeight = ginv((diag(ncol(H))/PenaltyP) + (t(H) %*% H)) %*% t(H) %*% Target  # implementation with regularization factor
  }
  
  Y=as.vector(unlist(t(H %*% outputWeight)))                    #   Y: the actual output of the training data
  #for the os-ELM
  #P0 <- ginv(t(H)%*%H)
  #rm(H)
  
  list(inputWeight=inputWeight,
       biasofHN=biasofHN,
       matrixBeta=outputWeight,
       #matrixP=P0, 
       predictionTrain=Y,
       rangeweight=rangew,
       activation=activation,
       outputBias=outputBias)
}#end function elm.optmization

Elm.predict <- function(TrainedElm, X.fit){
  X.fit <- t(X.fit)
  NumberofData=ncol(X.fit)
  ################## Valid ##########################
  
  B <- TrainedElm$matrixBeta
  inputWeight <- TrainedElm$inputWeight
  biasofHN <- TrainedElm$biasofHN
  
  tempH=inputWeight%*%X.fit
  ind=matrix(1,1,NumberofData)
  BiasMatrix=biasofHN%*%ind              #   Extend the bias matrix biasofHN to match the demention of H
  #clean a memory
  rm(ind) 
  tempH=tempH + BiasMatrix
  #clean a memory
  rm(BiasMatrix)
  
  #%%%%%%%% Sigmoid 
  switch(TrainedElm$activation,
         'TANH' = {H=t(tanh(tempH))},
        {H= t(1 / (1 + exp(-tempH)))}
  )
  rm(tempH)
  
  if(TrainedElm$outputBias){
    return(as.vector(unlist(t(cbind(1,H) %*% B))))    #   %   TY: the actual output of the testing data with output HN
  }else{
    return(as.vector(unlist(t(H %*% B))))             #   %   TY: the actual output of the testing data
  }
}#end function elm.Predict

elm.ensemble <- function(X.fit, Y.fit, X.test, ensem= 20, NumberofHiddenNeurons=10){
  ######### dividing valid set ############################
  nTrain = length(Y.fit)  
  resultEsTrain<-matrix(NA,nrow(X.fit),ensem) 
  resultEsTest<-matrix(NA,nrow(X.test),ensem)
    
  ############################### generates the ensemble
  for (e in 1:ensem) {
    elm.Trained <- elm.optmization(X.fit,Y.fit, 0, NumberofHiddenNeurons)
    resultEsTrain[,e] = elm.Trained$PredictionTrain
    resultEsTest[,e] = elm.Predict(elm.Trained,X.test)
  }# end for ensemble
    
  return (list(Trained=resultEsTrain,Tested=resultEsTest))
}# end function elm.ensemble

Elm.cross.valid <- function(X.fit, Y.fit, Number.hn, n.blocks=5,autorangeweight=FALSE){
  n.cases = length(Y.fit)
  index.block <- xval.buffer(n.cases, n.blocks)
  
  pred.ens.valid <- matrix(NA, n.cases, 1)
  
  for(nb in 1:n.blocks){
    fit.elm <- Elm.optmization(X.fit[index.block[[nb]]$train,,drop=FALSE], Y.fit[index.block[[nb]]$train,,drop=FALSE],
                               Number.hn=Number.hn,autorangeweight=autorangeweight)               
    pred.ens.valid[index.block[[nb]]$valid,1] = Elm.predict(fit.elm, X.fit[index.block[[nb]]$valid,,drop=FALSE])
  }#end blocks
  return(pred.ens.valid)
}# end ensemble


Elm.search.hn <- function(X.fit, Y.fit, n.ensem= 10, n.blocks=5, 
                          ErrorFunc=RMSE, percentValid=20,maxHiddenNodes = NULL,
                          Trace=TRUE, autorangeweight=FALSE){
  ###################### ajustando as informacoes do conjunto  #############################
  acceleration <- 1.51
  candidates <- c((-1/4*acceleration),0,(1/acceleration),acceleration)
  currentPoint <- 3
  stepSize <- 2
  cand.error.train <- rep(Inf,4) #training error of the candidates
  cand.error.valid <- rep(Inf,4) #training RMSE of the candidates
  bestHN <- c(Inf,Inf,Inf) #best HN, training error and validation error
  
  ######### dividing valid set ############################
  n.cases = length(Y.fit)
  if (n.blocks!=1){
    index.block <- xval.buffer(n.cases, n.blocks)
  }else{
    indValid <- n.cases-round((n.cases*(percentValid/100)))
  }
  
  if(is.null(maxHiddenNodes)){
    maxHiddenNodes <- (n.cases-1)
  }

  TOP <- TRUE
  while(TOP) {
    if (n.blocks!=1){
      pred.ens.valid <- matrix(NA, n.cases, n.ensem)
    }else{
      pred.ens.train <- matrix(NA, indValid, n.ensem)
      pred.ens.valid <- matrix(NA, (n.cases-indValid), n.ensem)
    }
  
    #filling the actual point without calculate it again
    cand.error.train[2] = bestHN[2]
    cand.error.valid[2] = bestHN[3]
    
    ############################### Random generate input weights inputWeight (w_i) and biases biasofHN (b_i) of hidden neurons
    for (ic in c(1,3,4)){ #1:length(candidates)
      for (e in 1:n.ensem) {
        n.hidden.can <- round(currentPoint + stepSize * candidates[ic])
        
        #constraint
        #testHiddenNeurons =max(1,testHiddenNeurons)
        
        if(n.blocks!=1){
            pred.ens.valid[,e] = Elm.cross.valid(X.fit,Y.fit,n.hidden.can,n.blocks,autorangeweight)
        }else{
            fit.elm <- Elm.optmization(X.fit[(1:indValid),,drop=FALSE],Y.fit[(1:indValid),drop=FALSE], 
                                       Number.hn=n.hidden.can,autorangeweight=autorangeweight) 
            pred.ens.train[,e] = fit.elm$predictionTrain
            pred.ens.valid[,e] = Elm.predict(fit.elm, X.fit[((indValid+1):nTrain),,drop=FALSE])
        }
      }# end ensemble
      
      if(n.blocks!=1){
        cand.error.train[ic] = cand.error.valid[ic] = ErrorFunc(Y.fit,rowMeans(pred.ens.valid))
      }else{
        cand.error.train[ic] = ErrorFunc(Y.fit[(1:indValid),drop=FALSE],rowMeans(pred.ens.train))
        cand.error.valid[ic] = ErrorFunc(Y.fit[((indValid+1):nTrain),drop=FALSE],rowMeans(pred.ens.valid))        
      }
    }#end candidates
    
    bestSolution <- which.min(cand.error.valid)
    if((bestHN[3] > cand.error.valid[bestSolution]) & (bestHN[2] > cand.error.train[bestSolution])){
      if(bestHN[1] == max(1,round(currentPoint + stepSize * candidates[bestSolution]))){
        cat('REPITIU??: \n')
        break
      } 
        
      bestHN[1] = max(1,round(currentPoint + stepSize * candidates[bestSolution]))
      bestHN[2] = cand.error.train[bestSolution]
      bestHN[3] = cand.error.valid[bestSolution]
      if(Trace){
        cat('hn: ', bestHN[1], ' step:',stepSize,
            'RMSE Train:', cand.error.train[bestSolution], 
            'RMSE Valid:', cand.error.valid[bestSolution], ' bs:', bestSolution, ' Cand:', 
            round(currentPoint + stepSize * candidates), '\n')
      }
      
      if (bestSolution !=2){
        currentPoint = bestHN[1]
        stepSize <- max(1, round(stepSize*candidates[bestSolution]))  
        #cat('??: ',stepSize * candidates[bestSolution],' forte:',bestSolution,'\n')
      }
      if(bestHN[1] >= maxHiddenNodes) break
    }else{
        break
    }#end if bestHN
  }#end while
  
  if(Trace){
    cat('Best Solution: ',bestHN[1],'\n')
  }
  
  return (bestHN[1])
}# end function Elm.searchNeuronsHC


#################################################################################################
#                                                                                               #
#                                       ELM Ends Here                                           #
#                                                                                               #
#################################################################################################

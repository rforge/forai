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
#     Aranildo R. Lima  <arodrigu@eos.ubc.ca>                                  #
#                                                                              #
# References:                                                                  #
#       G.-B. Huang, Q.-Y. Zhu, C.-K. Siew (2006)                              #
#       Extreme learning machine: Theory and applications                      #
#       Neurocomputing 70 (2006) 489-501                                       #
#                                                                              #
#       Lima, A.R.; A.J. Cannon and W.W. Hsieh.                                #
#       Nonlinear Regression In Environmental Sciences Using Extreme Learning  #
#       #Machines. Submited to: Environmental Modelling and Software -         #
#       ELSEVIER (submitted 2014/2/3).                                         #
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
Elm.train <- 
  function(X.fit, Y.fit, Number.hn=10, autorangeweight=FALSE, rangeweight=1, 
           activation='TANH',outputBias = FALSE, rangebias=1){
  ############## loading the values ###################
  X.fit <- t(X.fit)
  Target <- Y.fit
  n.InputNeurons=nrow(X.fit)
  n.TrainingData=ncol(X.fit)
  
  ####### automatic range of the weights based on the number of predictors  
  if(autorangeweight){
      switch(activation,
             'TANH' = {a=1},
                      {a=2}
              )
      rangeweight <- a*(n.InputNeurons)^-.5 
                      #sqrt(6)/sqrt(Number.hn+n.InputNeurons)
      rangebias <- a
  }

  ################## Random generate input weights inputWeight (w_i) and biases biasofHN (b_i) of hidden neurons
  inputWeight=matrix(runif(Number.hn*n.InputNeurons, min=-1, max=1),Number.hn, n.InputNeurons)*rangeweight
  biasofHN=runif(Number.hn, min=-1, max=1)*rangebias
  
  tempH=inputWeight%*%X.fit
  ind=matrix(1,1,n.TrainingData)
  BiasMatrix=biasofHN%*%ind      # Extend the bias matrix biasofHN to match the demention of H
  tempH=tempH+BiasMatrix
  
  #cleaning memory
  rm(ind) 
  rm(BiasMatrix)
  
  ############# activaction function ######################
  switch(activation,
         'TANH' = {H=t(tanh(tempH))},
         {H= t(1 / (1 + exp(-tempH)))}
         )
  rm(tempH)   #cleaning memory

  #bias parameter in the output layer
  if(outputBias){
    H <- cbind(1,H)    
  }

#  if (PenaltyP==0)
#  {
    outputWeight=(ginv(H) %*% Target)         # implementation without regularization factor //refer to 2006 Neurocomputing paper
#  }else{
    #outputWeight = ginv((diag(ncol(H))/PenaltyP) + (t(H) %*% H)) %*% t(H) %*% Target  # implementation with regularization factor
#  }
  
  Y=as.vector(unlist(t(H %*% outputWeight)))  #   Y: the actual output of the training data
  #for the os-ELM
  P0 <- ginv(t(H)%*%H)
  #rm(H)
  
  list(inputWeight=inputWeight,
       biasofHN=biasofHN,
       matrixBeta=outputWeight,
       matrixP=P0, 
       predictionTrain=Y,
       rangeweight=rangeweight,
       activation=activation,
       outputBias=outputBias,
       rangebias = rangebias)
}#end function elm.optmization

Elm.predict <- function(TrainedElm, X.fit){
  ############## loading the values ###################
  X.fit <- t(X.fit)
  NumberofData=ncol(X.fit)
  
  ###############   
  B <- TrainedElm$matrixBeta
  inputWeight <- TrainedElm$inputWeight
  biasofHN <- TrainedElm$biasofHN
  
  tempH=inputWeight%*%X.fit
  ind=matrix(1,1,NumberofData)
  BiasMatrix=biasofHN%*%ind              #   Extend the bias matrix biasofHN to match the demention of H
  tempH=tempH + BiasMatrix
  
  #cleaning memory
  rm(ind) 
  rm(BiasMatrix)
  
  #%%%%%%%% Sigmoid 
  switch(TrainedElm$activation,
         'TANH' = {H=t(tanh(tempH))},
        {H= t(1 / (1 + exp(-tempH)))}
  )
  rm(tempH) #cleaning memory
  
  if(TrainedElm$outputBias){
    return(as.vector(unlist(t(cbind(1,H) %*% B))))    #   output using the output bias
  }else{
    return(as.vector(unlist(t(H %*% B))))             #   output without the output bias
  }
}#end function elm.Predict

Elm.cross.valid <- function(X.fit, Y.fit, Number.hn=10, n.blocks=5, returnmodels = FALSE, 
                            autorangeweight=FALSE, rangeweight=1, 
                            activation='TANH',outputBias = FALSE, rangebias=1){
  #loading indices
  n.cases = length(Y.fit)
  index.block <- xval.buffer(n.cases, n.blocks)
  
  #creating the list of elms
  t.elmf <- list()
  length(t.elmf) <- n.blocks
  
  pred.ens.valid <- matrix(NA, n.cases, 1)
  
  for(nb in 1:n.blocks){
    t.elmf[[nb]] <- Elm.train(X.fit[index.block[[nb]]$train,,drop=FALSE], Y.fit[index.block[[nb]]$train,,drop=FALSE],
                               Number.hn=Number.hn,autorangeweight=autorangeweight, rangeweight=rangeweight, 
                               activation=activation,outputBias = outputBias, rangebias= rangebias)               
    pred.ens.valid[index.block[[nb]]$valid,1] = Elm.predict(t.elmf[[nb]], X.fit[index.block[[nb]]$valid,,drop=FALSE])
  }#end blocks
  
  if (returnmodels == FALSE){
    return(pred.ens.valid)
  }else{
    return(list(predictionTrain =  pred.ens.valid, trained.elms = t.elmf))
  }
}# end ensemble


Elm.search.hc <- function(X.fit, Y.fit, n.ensem= 10, n.blocks=5, 
                          ErrorFunc=RMSE, PercentValid=20,maxHiddenNodes = NULL,
                          Trace=TRUE, autorangeweight=FALSE, rangeweight=1, 
                          activation='TANH',outputBias = FALSE, rangebias=1){
  ###################### ajustando as informacoes do conjunto  #############################
  acceleration <- 1.51
  candidates <- c((-1/4*acceleration),0,(1/acceleration),acceleration)
  currentPoint <- 2
  stepSize <- 2
  cand.error.train <- rep(Inf,4) #training error of the candidates
  cand.error.valid <- rep(Inf,4) #training RMSE of the candidates
  bestHN <- matrix(Inf,1,3) #best HN, training error and validation error
  colnames(bestHN) <- c('HN',"TRAIN","VALID")
  
  ######### dividing valid set ############################
  n.cases = length(Y.fit)
  if (n.blocks==1){
    indValid <- n.cases-round((n.cases*(PercentValid/100)))
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
    cand.error.train[2] = bestHN[,'TRAIN']
    cand.error.valid[2] = bestHN[,'VALID']
    
    ############################### Testing candidates  #################################
    for (ic in c(1,3,4)){ #1:length(candidates)
      for (e in 1:n.ensem) {
        n.hidden.can <- round(currentPoint + stepSize * candidates[ic])
        
        if(n.blocks!=1){
            pred.ens.valid[,e] = Elm.cross.valid(X.fit,Y.fit,n.hidden.can,n.blocks,
                                                 autorangeweight=autorangeweight, rangeweight=rangeweight, 
                                                 activation=activation, outputBias=outputBias, rangebias=rangebias)
        }else{
            fit.elm <- Elm.train(X.fit[(1:indValid),,drop=FALSE],Y.fit[(1:indValid),drop=FALSE], 
                                       Number.hn=n.hidden.can, 
                                 autorangeweight=autorangeweight, rangeweight=rangeweight, 
                                 activation=activation, outputBias=outputBias, rangebias=rangebias) 
            pred.ens.train[,e] = fit.elm$predictionTrain
            pred.ens.valid[,e] = Elm.predict(fit.elm, X.fit[((indValid+1):n.cases),,drop=FALSE])
        }
      }# end ensemble
      
      if(n.blocks!=1){
        cand.error.train[ic] = cand.error.valid[ic] = ErrorFunc(Y.fit,rowMeans(pred.ens.valid))
      }else{
        cand.error.train[ic] = ErrorFunc(Y.fit[(1:indValid),drop=FALSE],rowMeans(pred.ens.train))
        cand.error.valid[ic] = ErrorFunc(Y.fit[((indValid+1):n.cases),drop=FALSE],rowMeans(pred.ens.valid))        
      }
    }#end candidates
    
    bestSolution <- which.min(cand.error.valid)
    if((bestHN[,"VALID"] > cand.error.valid[bestSolution]) & (bestHN[,"TRAIN"] > cand.error.train[bestSolution])){
      if(bestHN[,"HN"] == max(1,round(currentPoint + stepSize * candidates[bestSolution]))){
        #cat('Best solution : \n')
        break
      } 
      
      bestHN[,"HN"] = max(1,round(currentPoint + stepSize * candidates[bestSolution]))
      bestHN[,"TRAIN"] = cand.error.train[bestSolution]
      bestHN[,"VALID"] = cand.error.valid[bestSolution]
      if(Trace){
        cat('hn: ', bestHN[,"HN"], ' step:',stepSize,
            'RMSE Train:', cand.error.train[bestSolution], 
            'RMSE Valid:', cand.error.valid[bestSolution], ' bs:', bestSolution, ' Cand:', 
            round(currentPoint + stepSize * candidates), '\n')
      }
      
      if (bestSolution !=2){
        currentPoint = bestHN[,"HN"]
        stepSize <- max(1, round(stepSize*candidates[bestSolution]))  
        #cat('??: ',stepSize * candidates[bestSolution],' forte:',bestSolution,'\n')
      }else{
        break
      }
      if(bestHN[,"HN"] >= maxHiddenNodes) break
    }else{
        break
    }#end if bestHN
  }#end while
  
  if(Trace){
    cat('Final Best Solution: ',bestHN[,"HN"],'\n')
  }
  
  return (bestHN[1])
}# end function Elm.searchNeuronsHC

#################################################################################################
#                                                                                               #
#                                       ELM Ends Here                                           #
#                                                                                               #
#################################################################################################

#################################################################################################
#                                                                                               #
#                                       OL-ELM Starts Here                                           #
#                                                                                               #
#################################################################################################
Olelm.update <- function(TrainedElm, ppredictors, ptarget){
  ############# preparin the set   ######################
  predictors <- t(ppredictors)
  Target <- ptarget
  n.TrainingData=ncol(predictors)
  
  ############# reading from the previous ELM   ######################
  inputWeight = TrainedElm$inputWeight
  biasofHN = TrainedElm$biasofHN
  tempH = TrainedElm$inputWeight%*%predictors   # Calculate the partial hidden layer output matrix
  
  ind=matrix(1,1,n.TrainingData)
  biasMatrix = TrainedElm$biasofHN%*%ind              #   Extend the bias matrix biasofHN to match the demention of H
  rm(ind) #limpando a memoria
  tempH=tempH + biasMatrix
  rm(biasMatrix)
  
  ############# adding chunk  ######################
  switch(TrainedElm$activation,
         'TANH' = {H=t(tanh(tempH))},
         'RECT' = {H=t(log(1+exp(tempH)))},
{H= t(1 / (1 + exp(-tempH)))}
  )
rm(tempH)

if (TrainedElm$outputBias){
  new.Matrix <- olelm.update.beta(cbind(1,H), TrainedElm$matrixP, TrainedElm$matrixBeta, Target)
}else{
  new.Matrix <- olelm.update.beta(H, TrainedElm$matrixP, TrainedElm$matrixBeta, Target)
}

return(list(inputWeight=inputWeight,
            biasofHN=biasofHN,
            matrixBeta=new.Matrix$matrixBeta,
            matrixP=new.Matrix$matrixP,
            PredictionTrain=new.Matrix$PredictionTrain,
            rangeweight=TrainedElm$rangeweight,
            activation=TrainedElm$activation,
            outputBias=TrainedElm$outputBias,
            rangebias=TrainedElm$outputBias))
}#end function olelm.update

olelm.update.beta <- function(H0, P0, B0, Target){
  ############# preparin the set   ######################
  n.TrainingData <- nrow(Target)
  
  #Calculate the output weight beta
  if (n.TrainingData > 1){
    identityMatrix <- diag(nrow(H0))
    inverseStep <- ginv(identityMatrix+(H0%*%P0%*%t(H0)))
    rm(identityMatrix)
    newP = P0 - (P0%*%t(H0))%*%inverseStep%*%(H0%*%P0)
    rm(inverseStep)
  }else{
    denominator <- P0%*%t(H0)%*%H0%*%P0
    numerator <- 1+H0%*%P0%*%t(H0)
    
    newP = P0 - (denominator/as.numeric(numerator))
    rm(denominator)
  }
  newBeta = B0 + (newP%*%t(H0))%*%(Target-H0%*%B0) 
  
  Y=as.vector(unlist(t(H0 %*% newBeta)))                    #   Y: the actual output of the training data
  
  return(list(matrixBeta=newBeta,
              matrixP=newP,
              PredictionTrain=Y))
}#end function olelm.update


#################################################################################################
#                                                                                               #
#                                       OL-ELM Ends Here                                        #
#                                                                                               #
#################################################################################################

# Elm.search.bias <- function(rangebias,Number.hn,  
#                             X.fit, Y.fit, n.ensem= 10, n.blocks=5, 
#                             ErrorFunc=RMSE, PercentValid=20,
#                             Trace=TRUE, 
#                             activation='TANH',outputBias = FALSE){
#   
#   ######### dividing valid set ############################
#   if (all(rangebias >0)){
#     n.cases = length(Y.fit)
#     if (n.blocks==1){
#       indValid <- n.cases-round((n.cases*(PercentValid/100)))
#     }
#     
#       if (n.blocks!=1){
#         pred.ens.valid <- matrix(NA, n.cases, n.ensem)
#       }else{
#         pred.ens.train <- matrix(NA, indValid, n.ensem)
#         pred.ens.valid <- matrix(NA, (n.cases-indValid), n.ensem)
#       }
#       #B=100
#       #A=2
#     rangebias <- abs(rangebias)
#       #Number.hn <- round((rangebias[1] * (B-A+1))+1)
#       ############################### Testing rangeweight  #################################
#       for (e in 1:n.ensem) {
#         if(n.blocks!=1){
#           pred.ens.valid[,e] = Elm.cross.valid(X.fit,Y.fit,Number.hn,n.blocks,
#                                                autorangeweight=FALSE, rangeweight=rangebias[1], 
#                                                activation=activation, outputBias=outputBias, rangebias=rangebias[2])
#         }else{
#           fit.elm <- Elm.train(X.fit[(1:indValid),,drop=FALSE],Y.fit[(1:indValid),drop=FALSE], 
#                                Number.hn=Number.hn, 
#                                autorangeweight=FALSE, rangeweight=rangebias[1], 
#                                activation=activation, outputBias=outputBias, rangebias=rangebias[2]) 
#           pred.ens.train[,e] = fit.elm$predictionTrain
#           pred.ens.valid[,e] = Elm.predict(fit.elm, X.fit[((indValid+1):n.cases),,drop=FALSE])
#         }
#       }
#         
#         if(n.blocks!=1){
#           cand.error.train = cand.error.valid = ErrorFunc(Y.fit,rowMeans(pred.ens.valid))
#         }else{
#           cand.error.train = ErrorFunc(Y.fit[(1:indValid),drop=FALSE],rowMeans(pred.ens.train))
#           cand.error.valid = ErrorFunc(Y.fit[((indValid+1):n.cases),drop=FALSE],rowMeans(pred.ens.valid))        
#         }
#       
#     if(Trace){
#       cat('Train: ',cand.error.train, " - Valid: ",cand.error.train,'\n')
#     }
#     
#     round(cand.error.valid,4)
#   }else{
#     a = 10
#     a
#   }
# }# end function Elm.searchNeuronsHC
# 
# Elm.search.auto <- function(X.fit, Y.fit, n.ensem= 10, n.blocks=5, 
#                             ErrorFunc=RMSE, PercentValid=20,maxHiddenNodes = NULL,
#                             Trace=TRUE, 
#                             activation='TANH',outputBias = FALSE, 
#                             returnmodels = FALSE){
#   ###################### ajustando as informacoes do conjunto  #############################
#   acceleration <- 1.51
#   candidates <- c((-1/2*acceleration),0,(1/acceleration),acceleration)
#   currentPoint <- 3
#   stepSize <- 2
#   
#   cand.error.train <- rep(Inf,4) #training error of the candidates
#   cand.error.valid <- rep(Inf,4) #training RMSE of the candidates
#   
#   bestHN <- matrix(Inf,1,5) #best HN, training error and validation error
#   colnames(bestHN) <- c('HN',"TRAIN","VALID","RANGE","BIAS")
#   
#   ######### dividing valid set ############################
#   n.cases = length(Y.fit)
#   if (n.blocks==1){
#     indValid <- n.cases-round((n.cases*(PercentValid/100)))
#   }
#   
#   if(is.null(maxHiddenNodes)){
#     maxHiddenNodes <- (n.cases-1)
#   }
#   
#   switch(activation,
#          'TANH' = {a=1},
# {a=2}
#   )
# rangeweight <- a*(ncol(X.fit))^-.5 
# 
# currentPoint.range <- rangeweight
# stepSize.range <- rangeweight
# currentPoint.bias <- rangeweight
# stepSize.bias <- rangeweight
# 
# bestHN[,'RANGE'] = bestHN[,'BIAS'] = rangeweight
# bestHN[,'HN'] = currentPoint
# 
# global.go <- TRUE
# while(global.go) {
#   if (n.blocks!=1){
#     pred.ens.valid <- matrix(NA, n.cases, n.ensem)
#   }else{
#     pred.ens.train <- matrix(NA, indValid, n.ensem)
#     pred.ens.valid <- matrix(NA, (n.cases-indValid), n.ensem)
#   }
#   
#   cand.error.train[2] = bestHN[,'TRAIN']
#   cand.error.valid[2] = bestHN[,'VALID']
#   
#   ############################### Testing Bias  #################################
#   for (ic in c(1,3,4)){ #1:length(candidates)
#     for (e in 1:n.ensem) {
#       new.range <- (currentPoint.bias + stepSize.bias * candidates[ic])
#       if(new.range < 0 ) new.range = currentPoint.bias
#       
#       if(n.blocks!=1){
#         pred.ens.valid[,e] = Elm.cross.valid(X.fit,Y.fit,bestHN[,"HN"],n.blocks,
#                                              autorangeweight=FALSE, rangeweight=bestHN[,'RANGE'], 
#                                              activation=activation, outputBias=outputBias, rangebias=new.range)
#       }else{
#         fit.elm <- Elm.train(X.fit[(1:indValid),,drop=FALSE],Y.fit[(1:indValid),drop=FALSE], 
#                              Number.hn=bestHN[,"HN"], 
#                              autorangeweight=FALSE, rangeweight=bestHN[,'RANGE'], 
#                              activation=activation, outputBias=outputBias, rangebias=new.range) 
#         pred.ens.train[,e] = fit.elm$predictionTrain
#         pred.ens.valid[,e] = Elm.predict(fit.elm, X.fit[((indValid+1):n.cases),,drop=FALSE])
#       }
#     }# end ensemble
#     
#     if(n.blocks!=1){
#       cand.error.train[ic] = cand.error.valid[ic] = ErrorFunc(Y.fit,rowMeans(pred.ens.valid))
#     }else{
#       cand.error.train[ic] = ErrorFunc(Y.fit[(1:indValid),drop=FALSE],rowMeans(pred.ens.train))
#       cand.error.valid[ic] = ErrorFunc(Y.fit[((indValid+1):n.cases),drop=FALSE],rowMeans(pred.ens.valid))        
#     }
#   }#end candidates bias
#   
#   bestSolution <- which.min(cand.error.valid)
#   if((bestHN[,"VALID"] > cand.error.valid[bestSolution]) & (bestHN[,"TRAIN"] > cand.error.train[bestSolution])){
#     winner <- (currentPoint.bias + stepSize.bias * candidates[bestSolution])
#     if(winner < 0 ) winner = currentPoint.bias
#     
#     if(bestHN[,"BIAS"] != winner){
#       if(Trace){
#         cat('Bias: ', bestHN[,"BIAS"], ' step:',stepSize.bias,
#             'RMSE Train:', cand.error.train[bestSolution], 
#             ' bs:', bestSolution, ' Cand:', 
#             (currentPoint.bias + stepSize.bias * candidates), '\n')
#       }
#       if (bestSolution !=2){
#         if(winner < 0 ){
#           bestHN[,"BIAS"] = currentPoint.bias
#         }else{
#           bestHN[,"BIAS"] = winner
#           currentPoint.bias = bestHN[,"BIAS"]
#         } 
#         
#         winner <- (stepSize.bias*candidates[bestSolution])
#         if(winner > 0 ){
#           stepSize.bias <- (stepSize.bias*candidates[bestSolution])
#         }else{
#           stepSize.bias <- stepSize.bias/2
#         }
#         bestHN[,"TRAIN"] = cand.error.train[bestSolution]
#         bestHN[,"VALID"] = cand.error.valid[bestSolution]
#         global.go = TRUE
#       }
#     }
#   }
#   
#   #cat('??: ',stepSize * candidates[bestSolution],' forte:',bestSolution,'\n')
#   cand.error.train[2] = bestHN[,'TRAIN']
#   cand.error.valid[2] = bestHN[,'VALID']
#   
#   ############################### Testing Range  #################################
#   for (ic in c(1,3,4)){ #1:length(candidates)
#     for (e in 1:n.ensem) {
#       new.range <- (currentPoint.range + stepSize.range * candidates[ic])
#       if(new.range < 0 ) new.range = currentPoint.range
#       
#       if(n.blocks!=1){
#         pred.ens.valid[,e] = Elm.cross.valid(X.fit,Y.fit,bestHN[,"HN"],n.blocks,
#                                              autorangeweight=FALSE, rangeweight=new.range, 
#                                              activation=activation, outputBias=outputBias, rangebias=bestHN[,'BIAS'])
#       }else{
#         fit.elm <- Elm.train(X.fit[(1:indValid),,drop=FALSE],Y.fit[(1:indValid),drop=FALSE], 
#                              Number.hn=bestHN[,"HN"], 
#                              autorangeweight=FALSE, rangeweight=new.range, 
#                              activation=activation, outputBias=outputBias, rangebias=bestHN[,'BIAS']) 
#         pred.ens.train[,e] = fit.elm$predictionTrain
#         pred.ens.valid[,e] = Elm.predict(fit.elm, X.fit[((indValid+1):n.cases),,drop=FALSE])
#       }
#     }# end ensemble
#     
#     if(n.blocks!=1){
#       cand.error.train[ic] = cand.error.valid[ic] = ErrorFunc(Y.fit,rowMeans(pred.ens.valid))
#     }else{
#       cand.error.train[ic] = ErrorFunc(Y.fit[(1:indValid),drop=FALSE],rowMeans(pred.ens.train))
#       cand.error.valid[ic] = ErrorFunc(Y.fit[((indValid+1):n.cases),drop=FALSE],rowMeans(pred.ens.valid))        
#     }
#   }#end candidates range
#   
#   bestSolution <- which.min(cand.error.valid)
#   if((bestHN[,"VALID"] > cand.error.valid[bestSolution]) & (bestHN[,"TRAIN"] > cand.error.train[bestSolution])){
#     winner <- (currentPoint.range + stepSize.range * candidates[bestSolution])
#     if(winner < 0 ) winner = currentPoint.range
#     
#     if(bestHN[,"RANGE"] != winner){
#       if(Trace){
#         cat('Range: ', bestHN[,"RANGE"], ' step:',stepSize.range,
#             'RMSE Train:', cand.error.train[bestSolution], 
#             ' bs:', bestSolution, ' Cand:', 
#             (currentPoint.range + stepSize.range * candidates), '\n')
#       }
#       if (bestSolution !=2){
#         if(winner < 0 ){
#           bestHN[,"RANGE"] = currentPoint.range
#         }else{
#           bestHN[,"RANGE"] = winner
#           currentPoint.range = bestHN[,"RANGE"]
#         } 
#         
#         winner <- (stepSize.range*candidates[bestSolution])
#         if(winner > 0 ){
#           stepSize.range <- (stepSize.range*candidates[bestSolution])
#         }else{
#           stepSize.range <- stepSize.range/2
#         }
#         bestHN[,"TRAIN"] = cand.error.train[bestSolution]
#         bestHN[,"VALID"] = cand.error.valid[bestSolution]
#         global.go = TRUE
#       }
#     }
#   }
#   
#   
#   
#   #filling the actual point without calculate it again
#   cand.error.train[2] = bestHN[,'TRAIN']
#   cand.error.valid[2] = bestHN[,'VALID']
#   
#   ############################### Testing candidates  #################################
#   for (ic in c(1,3,4)){ #1:length(candidates)
#     for (e in 1:n.ensem) {
#       n.hidden.can <- round(currentPoint + stepSize * candidates[ic])
#       
#       if(n.blocks!=1){
#         pred.ens.valid[,e] = Elm.cross.valid(X.fit,Y.fit,n.hidden.can,n.blocks,
#                                              autorangeweight=FALSE, rangeweight=bestHN[,'RANGE'], 
#                                              activation=activation, outputBias=outputBias, rangebias=bestHN[,'RANGE'])
#       }else{
#         fit.elm <- Elm.train(X.fit[(1:indValid),,drop=FALSE],Y.fit[(1:indValid),drop=FALSE], 
#                              Number.hn=n.hidden.can, 
#                              autorangeweight=FALSE, rangeweight=bestHN[,'RANGE'], 
#                              activation=activation, outputBias=outputBias, rangebias=bestHN[,'RANGE']) 
#         pred.ens.train[,e] = fit.elm$predictionTrain
#         pred.ens.valid[,e] = Elm.predict(fit.elm, X.fit[((indValid+1):n.cases),,drop=FALSE])
#       }
#     }# end ensemble
#     
#     if(n.blocks!=1){
#       cand.error.train[ic] = cand.error.valid[ic] = ErrorFunc(Y.fit,rowMeans(pred.ens.valid))
#     }else{
#       cand.error.train[ic] = ErrorFunc(Y.fit[(1:indValid),drop=FALSE],rowMeans(pred.ens.train))
#       cand.error.valid[ic] = ErrorFunc(Y.fit[((indValid+1):n.cases),drop=FALSE],rowMeans(pred.ens.valid))        
#     }
#   }#end candidates
#   
#   bestSolution <- which.min(cand.error.valid)
#   global.go = FALSE
#   if((bestHN[,"VALID"] > cand.error.valid[bestSolution]) & (bestHN[,"TRAIN"] > cand.error.train[bestSolution])){
#     if(bestHN[,"HN"] != max(1,round(currentPoint + stepSize * candidates[bestSolution]))){
#       bestHN[,"HN"] = max(1,round(currentPoint + stepSize * candidates[bestSolution]))
#       bestHN[,"TRAIN"] = cand.error.train[bestSolution]
#       bestHN[,"VALID"] = cand.error.valid[bestSolution]
#       if(Trace){
#         cat('hn: ', bestHN[,"HN"], ' step:',stepSize,
#             'RMSE Train:', cand.error.train[bestSolution], 
#             'RMSE Valid:', cand.error.valid[bestSolution], ' bs:', bestSolution, ' Cand:', 
#             round(currentPoint + stepSize * candidates), '\n')
#       }
#       if(bestSolution !=2){
#         currentPoint = bestHN[,"HN"]
#         stepSize <- max(1, round(stepSize*candidates[bestSolution]))
#         global.go = TRUE
#       }
#     }
#   }
#   
#   if(bestHN[,"HN"] >= maxHiddenNodes) break
# }#end while
# 
# if(Trace){
#   cat('Final Best Solution: ',bestHN[,"HN"],'\n')
# }
# 
# return (bestHN)
# }# end function Elm.searchNeuronsHC
# Elm.search.range <- function(X.fit, Y.fit, Number.hn, rangeweight, rangebias, n.ensem= 10, n.blocks=5, 
#                              ErrorFunc=RMSE, PercentValid=20,
#                              Trace=TRUE, 
#                              activation='TANH',outputBias = FALSE){
#   ###################### ajustando as informacoes do conjunto  #############################
#   acceleration <- 1.51
#   candidates <- c((-1/acceleration),0,(1/acceleration),acceleration)
#   currentPoint <- rangeweight 
#   stepSize <- rangeweight
#   cand.error.train <- rep(Inf,4) #training error of the candidates
#   cand.error.valid <- rep(Inf,4) #training RMSE of the candidates
#   bestW =  matrix(Inf,1,3) #best HN, training error and validation error
#   colnames(bestW) <- c('W',"TRAIN","VALID")
#   
#   ######### dividing valid set ############################
#   n.cases = length(Y.fit)
#   if (n.blocks==1){
#     indValid <- n.cases-round((n.cases*(PercentValid/100)))
#   }
#   
#   TOP <- TRUE
#   while(TOP) {
#     if (n.blocks!=1){
#       pred.ens.valid <- matrix(NA, n.cases, n.ensem)
#     }else{
#       pred.ens.train <- matrix(NA, indValid, n.ensem)
#       pred.ens.valid <- matrix(NA, (n.cases-indValid), n.ensem)
#     }
#     
#     #filling the actual point without calculate it again
#     cand.error.train[2] = bestW[,'TRAIN']
#     cand.error.valid[2] = bestW[,'VALID']
#     
#     ############################### Testing rangeweight  #################################
#     for (ic in c(1,3,4)){ #1:length(candidates)
#       for (e in 1:n.ensem) {
#         rangeweight.can <- (currentPoint + stepSize * candidates[ic])
#         if(rangeweight.can < 0 ) rangeweight.can = currentPoint
#         
#         if(n.blocks!=1){
#           pred.ens.valid[,e] = Elm.cross.valid(X.fit,Y.fit,Number.hn,n.blocks,
#                                                autorangeweight=FALSE, rangeweight=rangeweight.can, 
#                                                activation=activation, outputBias=outputBias, rangebias=rangeweight.can)
#         }else{
#           fit.elm <- Elm.train(X.fit[(1:indValid),,drop=FALSE],Y.fit[(1:indValid),drop=FALSE], 
#                                Number.hn=Number.hn, 
#                                autorangeweight=FALSE, rangeweight=rangeweight.can, 
#                                activation=activation, outputBias=outputBias, rangebias=rangeweight.can) 
#           pred.ens.train[,e] = fit.elm$predictionTrain
#           pred.ens.valid[,e] = Elm.predict(fit.elm, X.fit[((indValid+1):n.cases),,drop=FALSE])
#         }
#       }# end ensemble
#       
#       if(n.blocks!=1){
#         cand.error.train[ic] = cand.error.valid[ic] = ErrorFunc(Y.fit,rowMeans(pred.ens.valid))
#       }else{
#         cand.error.train[ic] = ErrorFunc(Y.fit[(1:indValid),drop=FALSE],rowMeans(pred.ens.train))
#         cand.error.valid[ic] = ErrorFunc(Y.fit[((indValid+1):n.cases),drop=FALSE],rowMeans(pred.ens.valid))        
#       }
#     }#end candidates
#     
#     bestSolution <- which.min(cand.error.valid)
#     if((bestW[,"VALID"] > cand.error.valid[bestSolution]) & (bestW[,"TRAIN"] > cand.error.train[bestSolution])){
#       winner <-(currentPoint + stepSize * candidates[bestSolution])
#       if(winner < 0 ) winner = currentPoint
#       
#       if(bestW[,"W"] == winner){
#         #cat('Best solution : \n')
#         break
#       } 
#       
#       bestW[,"W"] = (currentPoint + stepSize * candidates[bestSolution])
#       bestW[,"TRAIN"] = cand.error.train[bestSolution]
#       bestW[,"VALID"] = cand.error.valid[bestSolution]
#       if(Trace){
#         cat('w: ', bestW[,"W"], ' step:',stepSize,
#             'RMSE Train:', cand.error.train[bestSolution], 
#             'RMSE Valid:', cand.error.valid[bestSolution], ' bs:', bestSolution, ' Cand:', 
#             (currentPoint + stepSize * candidates), '\n')
#       }
#       
#       if (bestSolution !=2){
#         currentPoint = bestW[,"W"]
#         stepSize <- stepSize + (stepSize*candidates[bestSolution])
#         #cat('??: ',stepSize * candidates[bestSolution],' forte:',bestSolution,'\n')
#       }else{
#         break
#       }
#     }else{
#       break
#     }#end if bestHN
#   }#end while
#   
#   if(Trace){
#     cat('Final w Best Solution: ',bestW[,"W"],'\n')
#   }
#   
#   return (list(rangeweight=bestW[,"W"],rangebias=bestW[,"W"]))
# }# end function Elm.searchNeuronsHC
# 
# Elm.bagging <- function(X.fit, Y.fit, n.ensem= 30, 
#                         ErrorFunc=RMSE, PercentValid=33, maxHiddenNodes = NULL,
#                         Trace=TRUE, autorangeweight=FALSE, rangeweight=1, 
#                         activation='TANH',outputBias = FALSE, rangebias=1){
#   
#   acceleration <- 1.51
#   candidates <- c((-1/4*acceleration),0,(1/acceleration),acceleration)
#   currentPoint <- 2
#   stepSize <- 2
#   cand.error.train <- rep(Inf,4) #training error of the candidates
#   bestHN <- matrix(Inf,1,2) #best HN, training error and validation error
#   colnames(bestHN) <- c('HN',"TRAIN")
#   
#   trained.candidates <- list()
#   length(trained.candidates) <- length(candidates)
#   trained.best <- NULL
#   
#   n.cases <- nrow(X.fit)
#   nTrain <- round(n.cases*(PercentValid/100))
#   
#   for(trial.oob in 1:30){
#     cases.specified <- list()
#     oob.specified <- c()
#     for(ens in 1:n.ensem){
#       block.length <- 1
#       cases.specified[[ens]] <- sample(nTrain)
#       #tsbootstrap(1:nrow(X.fit),
#       #        b=block.length,
#       #      type='block')
#       #oob.specified <- unique(c(oob.specified, which(!(1:n.cases %in%
#       #                                                         cases.specified[[ens]]))))
#     }
#     #if(length(oob.specified)==nrow(X.fit)) break
#   }
#   
#   if(is.null(maxHiddenNodes)){
#     maxHiddenNodes <- (n.cases-1)
#   }
#   
#   
#   
#   TOP <- TRUE
#   while(TOP) {
#     #filling the actual point without calculate it again
#     cand.error.train[2] = bestHN[,'TRAIN']
#     
#     ############################### Testing candidates  #################################
#     for (ic in c(1,3,4)){ #1:length(candidates)
#       trained.elm <- list()
#       length(trained.elm) <- n.ensem
#       pred.elm<-matrix(0,n.cases,n.ensem) #4-metodos e 4-erros
#       
#       for (e in 1:n.ensem) {
#         order.oob <- cases.specified[[ens]]#c(cases.specified[[e]],which(!(1:n.cases %in% cases.specified[[e]])))
#         #temp.percent <- PercentValid#length(which(!(1:nrow(X.fit) %in% cases.specified[[ea]])))/length(order.oob)
#         
#         n.hidden.can <- round(currentPoint + stepSize * candidates[ic])
#         
#         trained.elm[[e]] <- Elm.train(X.fit[order.oob,,drop=FALSE], Y.fit[order.oob,,drop=FALSE], 
#                                       Number.hn=n.hidden.can, 
#                                       autorangeweight=autorangeweight, rangeweight=rangeweight, 
#                                       activation=activation, outputBias=outputBias, rangebias=rangebias) 
#         pred.elm[,e]= Elm.predict(trained.elm[[e]],X.fit)  #predicting 
#       }# end ensemble
#       trained.candidates[[ic]] = trained.elm
#       cand.error.train[ic] = ErrorFunc(Y.fit,rowMeans(pred.elm))
#     }#end candidates
#     
#     bestSolution <- which.min(cand.error.train)
#     if(bestHN[,"TRAIN"] > cand.error.train[bestSolution]){
#       if(bestHN[,"HN"] == max(1,round(currentPoint + stepSize * candidates[bestSolution]))){
#         #cat('Best solution : \n')
#         break
#       } 
#       
#       bestHN[,"HN"] = max(1,round(currentPoint + stepSize * candidates[bestSolution]))
#       bestHN[,"TRAIN"] = cand.error.train[bestSolution]
#       trained.best <- trained.candidates[[bestSolution]]
#       if(Trace){
#         cat('hn: ', bestHN[,"HN"], ' step:',stepSize,
#             'RMSE Train:', cand.error.train[bestSolution], 
#             ' bs:', bestSolution, ' Cand:', 
#             round(currentPoint + stepSize * candidates), '\n')
#       }
#       
#       if (bestSolution !=2){
#         currentPoint = bestHN[,"HN"]
#         stepSize <- max(1, round(stepSize*candidates[bestSolution]))  
#         #cat('??: ',stepSize * candidates[bestSolution],' forte:',bestSolution,'\n')
#       }else{
#         break
#       }
#       if(bestHN[,"HN"] >= maxHiddenNodes) break
#     }else{
#       break
#     }#end if bestHN
#   }#end while
#   
#   if(Trace){
#     cat('Final Best Solution: ',bestHN[,"HN"],'\n')
#   }
#   
#   return (list(trained.elm=trained.best, train.rmse=bestHN[,'TRAIN'], bestHN=bestHN[,'HN']))
# }

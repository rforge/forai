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
      rangebias <- rangeweight
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
    if((bestHN[,"VALID"] > cand.error.valid[bestSolution]) & (bestHN[,"VALID"] > cand.error.train[bestSolution])){
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
  
  return (bestHN[,"HN"])
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


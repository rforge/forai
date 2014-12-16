library(MASS)

olmlr.optmization <- function(X.fit, Y.fit){

  H <- cbind(1,X.fit)
  Target <- Y.fit
  P0 <- ginv(t(H)%*%H)
  beta0 <- (P0%*%t(H)%*%Target)
  
  list(Matrixbeta=beta0,
       MatrixP=P0, 
       predictionTrain=as.vector(t(H %*% beta0)))
}

olmlr.update <- function(TrainedMLR, X.fit, Y.fit){
  H <- cbind(1, X.fit)
  Target <- Y.fit
  
  if (nrow(Y.fit) > 1){
    identityMatrix <- diag(nrow(H))
    inverseStep <- ginv(identityMatrix+(H%*%TrainedMLR$MatrixP%*%t(H)))
    rm(identityMatrix)
    P0 = TrainedMLR$MatrixP - (TrainedMLR$MatrixP%*%t(H))%*%inverseStep%*%(H%*%TrainedMLR$MatrixP)
    rm(inverseStep)
    
    beta0 = TrainedMLR$Matrixbeta + (P0%*%t(H))%*%(Target-H%*%TrainedMLR$Matrixbeta) 
  }else{
    denominator <- TrainedMLR$MatrixP%*%t(H)%*%H%*%TrainedMLR$MatrixP
    #numerator
    numerator <- 1+H%*%TrainedMLR$MatrixP%*%t(H)
    
    P0 = TrainedMLR$MatrixP - (denominator/as.numeric(numerator))
    rm(denominator)
    
    beta0 = TrainedMLR$Matrixbeta + (P0%*%t(H))%*%(t(Target)-H%*%TrainedMLR$Matrixbeta) 
  }
  
  list(Matrixbeta=beta0,
       MatrixP=P0, 
       predictionTrain=as.vector(t(H %*% beta0)))
}

olmlr.predict <- function(TrainedMLR, X.fit){
  H <- cbind(1, X.fit)
  
  return(t(H %*% TrainedMLR$Matrixbeta))
}
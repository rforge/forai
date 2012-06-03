rm(list=ls(all=TRUE)) 

library(qualV)
library(e1071)
library(FNN)
library(time)
library(doMC)
library(foreach)
registerDoMC()

x.train <- as.matrix(read.table("so2_train.inputs", header=FALSE))
y.train <- as.matrix(read.table("so2_train.targets", header=FALSE))
nTrain = nrow(y.train)

x.validMain <- as.matrix(read.table("so2_valid.inputs",header=FALSE))
y.validMain <- as.matrix(read.table("so2_valid.targets", header=FALSE))

################################# ALL PREDICTORS ##########################################################	

	#################### adjusting the set #########################
  x.fit <- x.train
  y.fit <- y.train

		 x.mn <- colMeans(x.fit)
		 x.sd <- apply(x.fit, 2, sd)
		 y.max <- max(y.fit)
		 
		 x.fit <- sweep(sweep(x.fit, 2, x.mn, '-'), 2, x.sd, '/')
		 x.valid <- sweep(sweep(x.validMain, 2, x.mn, '-'), 2, x.sd, '/')
		 
		 y.fit <- y.fit/y.max
		 y.valid <- y.validMain/y.max
		colnames(y.fit) <- c("Y")
		colnames(y.valid) <- c("Y")

	nNeighbor = 3
	kksvr <- knn.reg(x.fit,test=NULL,y=y.fit,k=nNeighbor)
	factor1 <- ((nTrain)^(1/5))*nNeighbor
	factor2 <- ((nTrain)^(1/5))*(nNeighbor-1)
	errorEst <- sqrt((factor1/factor2)*(kksvr$PRESS/nTrain))

	epsilCal <- 3*errorEst*(sqrt(log(nTrain)/nTrain))
	C =mean(y.fit) + 3*sd(y.fit)
	
  gama <- (2^(3:-15))[seq(1,19,by=2)]
  result<-matrix(0,1,5) #
  colnames(result) <- c("MAE","MSE", "GAMMA","EPSIL","C")

  res <- foreach (g = gama, .combine=cbind) %dopar% {
		sv <- svm(x.fit,y.fit,kernel="radial",gamma=g, epsilon=epsilCal, cost=C,cross=5)
		p.svm <- predict(sv, x.valid)

    MSE(y.valid, p.svm)
	}

  fGama <-  as.numeric(gama[which.min(res)])

  sv <- svm(x.fit,y.fit,kernel="radial",gamma=fGama, epsilon=epsilCal, cost=C,cross=5)
  p.svm <- predict(sv, x.valid)

  result[1,1] = MAE(y.valid,p.svm)
  result[1,2] = MSE(y.valid,p.svm)
  result[1,3] = fGama
  result[1,4] = epsilCal
  result[1,5] = C

  s=1
  newfilename <- paste(paste("GRID",s,sep=""),".csv",sep="")
  write.csv(p.svm, newfilename)
  
  newfilename <- paste(paste("ALL",s,sep=""),".csv",sep="")
  write.csv(result, newfilename)

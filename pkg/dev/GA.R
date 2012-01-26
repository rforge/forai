######################### Carrega as bibliotecas #######################
source('functionsGA.R')


ganow <- function(p_Geracoes, p_GeracaoSFilho, p_Mutacao, p_Peso, p_Escondida, 
			p_Lags, p_TamanhoPopulacao, p_InputTreino, p_TargetTreino,p_amostra,p_x.teste, p_y.teste){
################## Define os parametros a serem usados ######################
tamanhoGene = p_Lags*p_Escondida+p_Escondida*2+1
percentValid=20


	nTrain <- nrow(p_InputTreino)
	indiceValid <- nTrain-(nTrain*.2) 
	x.fit.train <- p_InputTreino[1:indiceValid,]
	x.fit.valid <- p_InputTreino[(indiceValid+1):nTrain,]
	y.fit.train <- p_TargetTreino[1:indiceValid]
	y.fit.valid <- p_TargetTreino[(indiceValid+1):nTrain]

	##################### Gerando Populacao ############################3
	valorOtimoPeso =  3/(sqrt(p_Escondida+1))
	comunidade <- matrix(runif(tamanhoGene*p_TamanhoPopulacao, min=-valorOtimoPeso, max=valorOtimoPeso),tamanhoGene, p_TamanhoPopulacao)

	arquivo="populacao"
	newfilename <- paste(paste(arquivo,p_amostra,sep=""),".csv",sep="")
	write.csv(comunidade, file=paste("./res/",newfilename,sep=""))

	####################### MSE #############
	tipoFitness= "MSE"
	cromossomo <- run.GA(p_Geracoes, p_GeracaoSFilho, p_Mutacao, p_Peso, p_Escondida, 
			tamanhoGene, p_TamanhoPopulacao, x.fit.train, y.fit.train,x.fit.valid,y.fit.valid,
			p_amostra,tipoFitness,comunidade,valorOtimoPeso)

	outputGA <- calcular.Output(cromossomo, p_Escondida, p_x.teste, p_y.teste, 0)

	arquivo= paste("GA",tipoFitness,sep="")
	newfilename <- paste(paste(arquivo,p_amostra,sep=""),".csv",sep="")
	write.csv(outputGA, file=paste("./set/",newfilename,sep=""))
	
	####################### ARV #############
	tipoFitness= "ARV"
	cromossomo <- run.GA(p_Geracoes, p_GeracaoSFilho, p_Mutacao, p_Peso, p_Escondida, 
			tamanhoGene, p_TamanhoPopulacao, x.fit.train, y.fit.train,x.fit.valid,y.fit.valid,
			p_amostra,tipoFitness,comunidade,valorOtimoPeso)

	outputGA <- calcular.Output(cromossomo, p_Escondida, p_x.teste, p_y.teste, 0)

	arquivo= paste("GA",tipoFitness,sep="")
	newfilename <- paste(paste(arquivo,p_amostra,sep=""),".csv",sep="")
	write.csv(outputGA, file=paste("./set/",newfilename,sep=""))

	####################### RMSE #############
	tipoFitness= "RMSE"
	cromossomo <- run.GA(p_Geracoes, p_GeracaoSFilho, p_Mutacao, p_Peso, p_Escondida, 
			tamanhoGene, p_TamanhoPopulacao, x.fit.train, y.fit.train,x.fit.valid,y.fit.valid,
			p_amostra,tipoFitness,comunidade,valorOtimoPeso)

	outputGA <- calcular.Output(cromossomo, p_Escondida, p_x.teste, p_y.teste, 0)

	arquivo= paste("GA",tipoFitness,sep="")
	newfilename <- paste(paste(arquivo,p_amostra,sep=""),".csv",sep="")
	write.csv(outputGA, file=paste("./set/",newfilename,sep=""))

	####################### MAE #############
	tipoFitness= "MAE"
	cromossomo <- run.GA(p_Geracoes, p_GeracaoSFilho, p_Mutacao, p_Peso, p_Escondida, 
			tamanhoGene, p_TamanhoPopulacao, x.fit.train, y.fit.train,x.fit.valid,y.fit.valid,
			p_amostra,tipoFitness,comunidade,valorOtimoPeso)

	outputGA <- calcular.Output(cromossomo, p_Escondida, p_x.teste, p_y.teste, 0)

	arquivo= paste("GA",tipoFitness,sep="")
	newfilename <- paste(paste(arquivo,p_amostra,sep=""),".csv",sep="")
	write.csv(outputGA, file=paste("./set/",newfilename,sep=""))

	####################### Theil #############
	tipoFitness= "THEIL"
	cromossomo <- run.GA(p_Geracoes, p_GeracaoSFilho, p_Mutacao, p_Peso, p_Escondida, 
			tamanhoGene, p_TamanhoPopulacao, x.fit.train, y.fit.train,x.fit.valid,y.fit.valid,
			p_amostra,tipoFitness,comunidade,valorOtimoPeso)

	outputGA <- calcular.Output(cromossomo, p_Escondida, p_x.teste, p_y.teste, 0)

	arquivo= paste("GA",tipoFitness,sep="")
	newfilename <- paste(paste(arquivo,p_amostra,sep=""),".csv",sep="")
	write.csv(outputGA, file=paste("./set/",newfilename,sep=""))

	
}# fim do for do numero de samples ou da funcao



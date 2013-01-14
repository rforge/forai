####################### Inicializa as funcoes ################################
#Draw.Individual <- function(ptotalSum, pfitnessPopulation, psizePopulation)
#Crossover.Average <- function(pindexFather, pindexMother, ppopulation)
#crossover.Maximo <- function(p_pai, p_mae, p_tamanhoGene, p_comunidade, p_peso)
#crossover.Minimo <- function(p_pai, p_mae, p_tamanhoGene, p_comunidade, p_peso)
#crossover.Duplo <- function(p_pai, p_mae, p_tamanhoGene, p_comunidade, p_peso)
#calcular.Fitness <- function(p_tipoFitness, p_Individuo, p_  p_TarTreino, p_SetTreino, p_Iteracao)
#source('erros.R')

######################### ######################### ######################### 
######################### roulete wheel ##############################
######################### ######################### ######################### 
Draw.Individual <- function(ptotalSum, pfitnessPopulation, psizePopulation)
{
  valueIndividual=0
  draw = runif(1, min=0, max=1)
  for (aa in 1:psizePopulation) {
    valueIndividual = valueIndividual + pfitnessPopulation[aa]
    
     if ((draw < (valueIndividual/ptotalSum)) && (aa==0))
             return (0)
    
     nextValue = p_FitnessPopulation[aa]
     if((draw > (valueIndividual/ptotalSum)) && 
       (draw < ((valueIndividual/ptotalSum)+ (nextValue/ptotalSum))))
             return (aa)
  }#end for
  return (psizePopulation)
}#end Draw.Individual

######################### ######################### ######################### 
######################### Cross Over functions ##############################
######################### ######################### ######################### 
Crossover.Mean <- function(pindexFather, pindexMother, ppopulation) {
      
  chromossomeFather <- ppopulation[,pindexFather]
  chromossomeMother <- ppopulation[,pindexMother]
  
  offspring <- apply(cbind(chromossomeFather,chromossomeMother),1,mean)
  return (offspring)
}#end crossover mean

Crossover.Maximum <- function(pindexFather, pindexMother, ppopulation, pweight) {
  sizeGene <- length(chromossomeFather)
  chromossomeFather <- ppopulation[,pindexFather]
  chromossomeMother <- ppopulation[,pindexMother]
  
	valueMaximum = max(max(chromossomeFather),max(chromossomeMother))

  offspring <- rep(0, times=sizeGene)
  for(s in 1:sizeGene)
  {
    maxValueFatherMother = max(chromossomeFather[s], chromossomeMother[s])
    offspring[s] = (maxValueFatherMother*(1- pweight)) + (maxValueFatherMother * pweight)
  }
  return (offspring)
}#Crossover.Maximum

Crossover.Minimum <- function(pindexFather, pindexMother, ppopulation, pweight) {
  sizeGene <- length(chromossomeFather)
  chromossomeFather <- ppopulation[,pindexFather]
  chromossomeMother <- ppopulation[,pindexMother]
  
  valueMinimum = min(min(chromossomeFather),min(chromossomeMother))

  offspring <- rep(0, times=sizeGene)
  for(s in 1:sizeGene)
  {
    minValueFatherMother = min(chromossomeFather[s], chromossomeMother[s])
    offspring[s] = (minValueFatherMother*(1- pweight)) + (minValueFatherMother * pweight)
  }
  return (offspring)
}#end Crossover.Minimum

crossover.Duplo <- function(p_pai, p_mae, p_tamanhoGene, p_comunidade, p_peso) {

       cromossomoPai <- p_comunidade[,p_pai]
       cromossomoMae <- p_comunidade[,p_mae]

       valorMinimo = min(min(cromossomoPai), min(cromossomoMae))
       valorMaximo = max(max(cromossomoPai), max(cromossomoMae))

       s=0
       filho <- rep(0, times=p_tamanhoGene)
       for(s in 1:p_tamanhoGene){
               filho[s]  = ((valorMinimo+valorMaximo)*(1-p_peso)
					+((cromossomoPai[s] + cromossomoMae[s])* p_peso))/2
	}
       return (filho)
}

######################### ######################### ######################### 
######################### funcoes de Calcular o Fitness #####################
######################### ######################### ######################### 
calcular.Fitness <- function(p_tipoFitness, p_Individuo, p_x.Train, p_y.Train){
      	
	evals.evecs <- try(svm(p_x.Train, p_y.Train,kernel="radial",gamma=exp(p_Individuo[1]), epsilon=exp(p_Individuo[2]), cost=exp(p_Individuo[3]))
		,silent=TRUE)

	if(class(evals.evecs)!='try-error'){
		nn <- evals.evecs 
	       	fitness <- switch(p_tipoFitness,
	      				"ARV"=1/(1+erro.ARV(p_y.Train, nn$fitted)), 
						"RMSE"=1/(1+RMSE(p_y.Train, nn$fitted)), 
						"MSE"=1/(1+MSE(p_y.Train, nn$fitted)),
						"MAE"=1/(1+MAE(p_y.Train, nn$fitted)), 
						1/(1+erro.THEIL(p_y.Train, nn$fitted)))
	}else{
		fitness = 0
	}
	return(fitness)
}

calcular.Fitness.Valid <- function(p_tipoFitness, p_Individuo, p_x.Train, p_y.Train, p_x.Valid, p_y.Valid){
      	
	evals.evecs <- try(svm(p_x.Train,p_y.Train,kernel="radial",gamma=exp(p_Individuo[1]), epsilon=exp(p_Individuo[2]), cost=exp(p_Individuo[3]))
		,silent=TRUE)

	if(class(evals.evecs)!='try-error'){
		nn <- evals.evecs
		gerado <- predict(nn, p_x.Valid)

	       	fitness <- switch(p_tipoFitness,
	      				"ARV"=1/(1+erro.ARV(p_y.Valid, gerado)), 
						"RMSE"=1/(1+RMSE(p_y.Valid, gerado)), 
						"MSE"=1/(1+MSE(p_y.Valid, gerado)),
						"MAE"=1/(1+MAE(p_y.Valid, gerado)), 
						1/(1+erro.THEIL(p_y.Valid, gerado)))
	}else{
		fitness=0
	}
	return(fitness)
}

######################### ######################### ######################### 
######################### funcoes de Calcular o Output ######################
######################### ######################### ######################### 

calcular.Output <- function(p_Individuo, p_x.Train, p_y.Train, p_x.Test){
      	nn <- svm(p_x.Train, p_y.Train,kernel="radial",gamma=exp(p_Individuo[1]), epsilon=exp(p_Individuo[2]), cost=exp(p_Individuo[3]))
	gerado <- predict(nn, p_x.Test)
	return(gerado)
}

run.GA <- function(p_Geracoes, p_GeracaoSFilho, p_Mutacao, p_Peso, 
			tamanhoGene, p_sizePopulation, x.fit.train, y.fit.train, x.fit.valid, y.fit.valid,
			p_amostra,tipoFitness,comunidade){
peso = p_Peso
probMutacao = p_Mutacao
geracaoMaxima = p_Geracoes
geracaosemFilhoMax = p_GeracaoSFilho
sizePopulation = p_sizePopulation
############## inicializacao variaveis ###############
geracaoSemFilho = 0
fitnessMedio =0	
mediaPop =0
fitnessComunidade <- c(1:sizePopulation)
fitnessComunidadeValid <- c(1:sizePopulation)

	for (a in 1:sizePopulation) {
		fitnessComunidade[a]= calcular.Fitness(tipoFitness, comunidade[,a],
					x.fit.train, y.fit.train)
		
		fitnessComunidadeValid[a]= calcular.Fitness.Valid(tipoFitness, comunidade[,a],
					x.fit.train, y.fit.train,x.fit.valid,y.fit.valid)
	}#fim for de gerar a populacao
	
	evolucao = c()
	for (g in 1:geracaoMaxima) {
		mediaPop = mean(fitnessComunidade[a])
	       	geracaoSemFilho = geracaoSemFilho +1
		       if (geracaoSemFilho > geracaosemFilhoMax)
		       	break

	##################### Sorteio Populacao ############################
	       	somaTotal= sum(fitnessComunidade)

		pai = sortear.ind(somaTotal,fitnessComunidade, sizePopulation)
		mae = sortear.ind(somaTotal,fitnessComunidade, sizePopulation)

	       	while (pai == mae){
		       	mae = sortear.ind(somaTotal,fitnessComunidade, sizePopulation)
		}

	################## Cross-Over ########################################
	       	filhoMedia <- crossover.Media(pai, mae, comunidade)
	       	filhoMaximo <- crossover.Maximo(pai, mae, tamanhoGene, comunidade, peso)
	       	filhoMinimo <- crossover.Minimo(pai, mae, tamanhoGene, comunidade, peso)
	       	filhoDuplo <- crossover.Duplo(pai, mae, tamanhoGene, comunidade, peso)

	       	fitnessMedia = calcular.Fitness(tipoFitness, filhoMedia, x.fit.train, y.fit.train)
	       	fitnessMaximo = calcular.Fitness(tipoFitness, filhoMaximo, x.fit.train, y.fit.train)
	       	fitnessMinimo = calcular.Fitness(tipoFitness, filhoMinimo, x.fit.train, y.fit.train)
	       	fitnessDuplo = calcular.Fitness(tipoFitness, filhoDuplo, x.fit.train, y.fit.train)

       #separando o melhor individuo do cross-over
	       	fitnessFilhos <- c(fitnessMedia,fitnessMaximo,fitnessMinimo,fitnessDuplo)

	       	melhorFilho <- switch(which.max(fitnessFilhos),
	      				"1"=filhoMedia, "2"=filhoMaximo, "3"=filhoMinimo, filhoDuplo)

	       	melhorFilhoFitness <- switch(which.max(fitnessFilhos),
							"1"=fitnessMedia, "2"=fitnessMaximo, "3"=fitnessMinimo, fitnessDuplo)

	       	melhorFilhoFitnessValid <- calcular.Fitness.Valid(tipoFitness, melhorFilho, x.fit.train, y.fit.train, x.fit.valid, y.fit.valid)
		

       #substituindo na populacao
	       	if((min(fitnessComunidade) < melhorFilhoFitness) && (min(fitnessComunidadeValid) < melhorFilhoFitnessValid)) {
		    	individuoFraco = which.min(fitnessComunidade)
		       	comunidade[,individuoFraco] = melhorFilho[]
		       	fitnessComunidade[individuoFraco] = melhorFilhoFitness
			fitnessComunidadeValid[individuoFraco] = melhorFilhoFitnessValid
			if(mediaPop < mean(fitnessComunidade[a])){		               	
				geracaoSemFilho =0
			}
	       	}

	########################## MUtacao ######################################
	       	mutTudo <- melhorFilho
	       	mutMeio <- melhorFilho
	       	multi1 <- melhorFilho

       #gerando o vetor p/ mutar 50% dos genes
	       	multi <- runif(tamanhoGene)
	       	multi[multi>0.5]=1
	       	multi[multi<1]=0

       #fazendo os 3 tipos de mutacao
	       	mutTudo = melhorFilho +  rnorm(tamanhoGene)
	       	mutMeio = melhorFilho +  (multi*rnorm(tamanhoGene))
	       	indiceMut = round(runif(1, 1, tamanhoGene))
	       	multi1[indiceMut] = melhorFilho[indiceMut] + rnorm(1)

	       	fitnessmutTudo = calcular.Fitness(tipoFitness, mutTudo, x.fit.train, y.fit.train)
	       	fitnessmutMeio = calcular.Fitness(tipoFitness, mutMeio, x.fit.train, y.fit.train)
	       	fitnessmulti1 = calcular.Fitness(tipoFitness, multi1, x.fit.train, y.fit.train)

	       #separando o melhor individuo da mutacao
	       	fitnessMutacao <- c(fitnessmutTudo,fitnessmutMeio,fitnessmulti1)
	       	melhorMutado <- switch(which.max(fitnessMutacao),
						"1"=mutTudo, "2"=mutMeio, multi1)

	      	melhorFitnessMutado <- switch(which.max(fitnessMutacao),
							"1"=fitnessmutTudo, "2"=fitnessmutMeio, fitnessmulti1)

	      	melhorFitnessMutadoValid <- calcular.Fitness.Valid(tipoFitness, multi1, x.fit.train, y.fit.train, x.fit.valid, y.fit.valid)

	      #substituindo na populacao
	      	if (runif(1) < probMutacao){
		       	individuoFraco = which.min(fitnessComunidade)
		       	comunidade[,individuoFraco] = melhorMutado[]
		       	fitnessComunidade[individuoFraco] = melhorFitnessMutado
			fitnessComunidadeValid[individuoFraco] = melhorFitnessMutadoValid
	       	}else{
		       	if(min(fitnessComunidade) < melhorFitnessMutado){
		          	individuoFraco = which.min(fitnessComunidade)
		               	comunidade[,individuoFraco] = melhorMutado[]
		               	fitnessComunidade[individuoFraco] = melhorFitnessMutado
				fitnessComunidadeValid[individuoFraco] = melhorFitnessMutadoValid
				if(mediaPop < mean(fitnessComunidade[a])){		               	
					geracaoSemFilho =0
				}
		       	}
	       	}
		evolucao <- c(evolucao, median(fitnessComunidade))		
	}# fim for da quantidade de geracoes

	arquivo= paste("evolucao",tipoFitness,sep="")
	newfilename <- paste(paste(arquivo,p_amostra,sep=""),".csv",sep="")
	write.csv(evolucao, file=paste("./res/",newfilename,sep=""))

	arquivo= paste("melhor",tipoFitness,sep="")
	newfilename <- paste(paste(arquivo,p_amostra,sep=""),".csv",sep="")
	write.csv(exp(comunidade[,which.max(fitnessComunidade)]), file=paste("./res/",newfilename,sep=""))

#plot(evolucao)
	return(comunidade[,which.max(fitnessComunidade)])

}#fim funcao

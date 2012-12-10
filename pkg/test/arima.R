rm(list=ls(all=TRUE)) 

library(tseries)
#library(gdata)
library(ts)
#library(forecast)
#Carregar os dados

diretorio = "/net/home/arodrig/so2/so2_train_normaliz"
inidata <- read.delim(paste(diretorio,".csv", sep=""),header=FALSE, sep=",")

z=read.table('ibovespa1.txt')
#z=read.table('nasdaq1.txt')
#z=read.table('dowjones1.txt')



f=as.ts(z[,1])
x=f
n=length(x)
x = f[1:(n-30)]


plot(x,type='l')
x[1:10]
#----------------------------------------------------------------------

#---------------------------------------------------------------------------
#TRANSFORMAÇÃO

# Na necessidade de estabilizar a variância realizamos a transformação 
#dos dados usando o teste de box-cox

library(MASS)
 
t1 <- 1:length(x) 

#criamos um modelo linear
y.lm <- lm(x~t1)
 
#rodamos o processo Box-Cox 
boxcox(y.lm,plotit=T)

#limitando a região de interesse
boxcox(y.lm,plotit=T,lam=seq(-1,0, 1/100))
boxcox(y.lm,plotit=T,lam=seq(0.5,1, 1/100000))
boxcox(y.lm,plotit=T,lam=seq(-0.8,-0.6, 1/100000))

#valor encontrado aproximado = -0.1387 ibovespa
#valor encontrado aproximado = 0.738  DJIA
#valor encontrado aproximado = -0.688  nasdaq

lambda = -0.1387
lambda = -0.688
lambda = 0.738

#realizando a transformação:

xtrans <- (x^(lambda) - 1)/(lambda)
write.table(xtrans,'xtrans.txt',row.names=F,col.names=F)

xtrans[1:10]
par(mfrow=c(2,1))
plot(x)
plot(xtrans)
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
#Augmented Dickey–Fuller Test

#Testa-se a hipótese nula de que a
#série não é estacionária (ou seja, possui raiz unitária) contra a alternativa
#de que a série é estacionária (não possui raiz unitária).
#observação se aparecer a mensagem warning , dizendo que o p-valor é menor
#a série é estacionária

require(fUnitRoots)
adfTest(xtrans.diff)
adfTest(x)



#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# OPERAÇÃO DIFERENÇA

# utilizando a função ndiffs() para determinar o numero de diferenças 
#necessarias pra tornar a serie estacionaria.

xtrans.diff=diff(x,ndifferences=ndiffs(x))


#xtrans.diff=diff(xtrans,ndifferences=ndiffs(xtrans))
#plot(xtrans.diff)

#adfTest(xtrans.diff)
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------

# OPERAÇÃO NORMALIZAÇÃO
mean(xtrans)

xtrans.norm=xtrans.diff-mean(xtrans.diff)

x.norm=x-mean(x)

mean(xtrans.norm)
#---------------------------------------------------------------------------


#----------------------------------------------------------------------------
# FUNÇÕES DE AUTOCORRELAÇÃO

#para comparar as fac's e facp's para a serie e para a serie diferenciada
par(mfrow=c(1,2))

plot(x,xlab='tempo',ylab='observações',main='',type='l')
plot(xtrans.norm,xlab='tempo',ylab='observações',main='Dados Transformados')
plot(xtrans.diff,xlab='tempo',ylab='observações',main='Dados Transformados diferenciada',type='l')

acf (x,lag.max=15,xlab='defasagem',ylab='fac',main='Autocorrelação')
pacf(x,lag.max=15,xlab='defasagem',ylab='facp',main='Autocorrelação Parcial')

acf (xtrans.diff,lag.max = 15, xlab='defasagem',ylab='fac',main='Autocorrelação')
pacf(xtrans.diff,lag.max = 15, xlab='defasagem',ylab='facp',main='Autocorrelação Parcial')



acf (xtrans.diff,lag.max = 15,xlab='defasagem',ylab='fac',main='1º diferença')
pacf(xtrans.diff,lag.max = 15,xlab='defasagem',ylab='facp',main='1º diferença')

acf (xtrans.norm,xlab='defasagem',ylab='fac',main='Dados diferenciados normalizados')
pacf(xtrans.norm,xlab='defasagem',ylab='facp',main='Dados diferenciados normalizados')

par(mfrow=c(1,2))
acf (xtrans,xlab='defasagem',ylab='fac',main='Dados Transformados')
pacf(xtrans,xlab='defasagem',ylab='facp',main='Dados Transformados')

#----------------------------------------------------------------------------


#----------------------------------------------------------------------------

# DIAGNOSTICO DO MODELO

#modelo p,d,q

k = 0:3
ind=0
xt=x.norm
n = length(x)
crit = matrix(0, nrow = length(k)^2, ncol = 6)
colnames(crit) = c("p", "d", "q", "AIC", "BIC","loglik")

for (p in k){
    #for (d in k) {
        for (q in k) {
            	ind = ind + 1
            	crit[ind,1]=p
            	crit[ind,2]=1
            	crit[ind,3]=q
            	arma = arima(xt, order = c(p, 1, q), include.mean = F,optim.control = list(maxit=500))
            	crit[ind, 4] = -2 * arma$loglik + 2 * (p+ q + 1 )
                  crit[ind, 5] = -2* arma$loglik + (p+q +1)* log(n)
			crit[ind,6] = arma$loglik
        }
    #}
}
crit
write.matrix(crit, file = "diagnostico_nasdaq.txt", sep = " ")
 


arma1 = arima(xt, order = c(2, 1, 2), include.mean = F)

tsdiag(arma1,gof.lag=15)


library(forecast)
plot(forecast.Arima(arma 1 , h=30, conf=c(95)),shaded=F,col=1)
fore = forecast.Arima(arma1 , h=560, conf=c(95))
#plot(fore, xlim=c(3100,3150),ylim=c(0.2,0.3))
plot(fore)


#---------------------------------------------------------------------------

#-------------------------------------------------------------------------------



ar(x, AIC=FALSE, order.max=1, method='yule-walker')
#pelo estimador de maxima verossimilhança ,method="ML", não é
necessário especifica-lo
m = arima(xt,order=c(3,1,1))
#Como primeiro verificação da adequação do modelo vamos usar a função
Box.test() para
#testar a hipotese de independencia residual. Por exemplo para testar
se as 3 primeiras autocorrelações são nulas pelo teste de Box-Pierce,
Box.test(m$residuals, lag=15, type='Box-Pierce')
Box.test(m$residuals, lag=15, type='Ljung-Box')



qchisq(0.95, 9, ncp=0, lower.tail = TRUE, log.p = FALSE)

# se Q(x) < ?2(k-p-q), é ruído branco

#---------------------------------------------------------------------------

#----------------Periodicidade----------------------------------------------

t = 1:n
c1 = cos(2*pi*t*1/n)
s1 = sin(2*pi*t*1/n)
c2 = cos(2*pi*t*2/n)
s2 = sin(2*pi*t*2/n)
creg = lm(x~c1+s1+c2+s2)
anova(creg)

scpec = abs(fft(x))^2/n

par(mfrow=c(1,1))

Ibovespa=xt

spectrum(Ibovespa)
spectrum(xtrans.diff)
per = spec.pgram(x, taper=0)

per = spec.pgram(xtrans.diff, taper=0, log="no")
per = spec.pgram(Ibovespa, taper=0, log="no")



periodo = scpec[1:(n/2)+1]




x = c(1,2,3,2,1)
t = 1:5
c1 = cos(2*pi*t*1/5)
s1 = sin(2*pi*t*1/5)
c2 = cos(2*pi*t*2/5)
s2 = sin(2*pi*t*2/5)
creg = lm(x~c1+s1+c2+s2)
anova(creg)




# Calculo do PERIODOGRAMA
P = abs(2*fft(x)/n)^2
f = 0:(n/2)/n
plot(f, P[1:(n/2+1)], type="l", xlab="frequência",ylab="periodograma")
abline(v=seq(0,.5,.02), lty="dotted")

# ver o periodograma acumulado
arma1 = arima(xt, order = c(3, 1, 2), include.mean = F)
tsdiag(arma1,gof.lag=15)
m = arima(xt,order=c(3,1,2))
resid=m$residuals
cpgram(resid, taper = 0.1,
       main = paste("Series: ARIMA (3,1,2)"),
       ci.col = "blue")


m = arima(xtrans,order=c(0,0,3))
resid=m$residuals
cpgram(resid, taper = 0.1,
       main = paste("Series: ", deparse(substitute(ts))),
       ci.col = "red")


m = arima(xt,order=c(3,1,3))
resid=m$residuals
cpgram(resid, taper = 0.1,
       main = paste("Series: ", deparse(substitute(ts))),
       ci.col = "brown")

m = arima(x,order=c(2,1,2))
resid=m$residuals
cpgram(resid, taper = 0.1,
       main = paste("Series: ", deparse(substitute(ts))),
       ci.col = "brown")


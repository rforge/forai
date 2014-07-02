\name{Elm.search.hc}
\alias{Elm.search.hc}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Search Number of Hidden Neurons Using Hill Climbing}
\description{Finds the number of hidden neurons using the hill climbing procedure.}
\usage{
Elm.search.hc(X.fit, Y.fit, n.ensem= 10, n.blocks=5, ErrorFunc=RMSE, PercentValid=20,
              maxHiddenNodes = NULL, Trace=TRUE, autorangeweight=FALSE, rangeweight=NULL, 
              activation='TANH',outputBias = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X.fit}{Data matrix (numeric) containing the input values (predictors) used to train the model.}
  \item{Y.fit}{Response vector (numeric) used to train the model.}
  \item{n.ensem}{Number of ensemble members. Default is \code{10}.}
  \item{n.blocks}{an integer specifying the desired number of cross-validation folds. Default is \code{5}.}
  \item{ErrorFunc}{Error function to be minimized. The default is the function \code{MSE} from package \code{qualV}        but the function could be customized.} 
  \item{PercentValid}{Percentage of the data reserved for validation (if \code{n.blocks < 2}. Default is \code{20}\%.}
  \item{maxHiddenNodes}{Maximum number of hidden nodes. Default is \code{NULL} which means \code{number of cases -1}.}
  \item{Trace}{If \code{TRUE}, information is printed during the running of \code{svres}. Default is \code{TRUE}.}
  \item{autorangeweight}{Option whether to use the automated range used for the weights. Default is \code{FALSE}.}
  \item{rangeweight}{Initial random weights on \code{[-rangeweight,rangeweight]}. The default is \code{NULL}}
  \item{activation}{Activation function of the hidden layer neurons. Available functions are: 'TANH' (default) and 'SIG'. }
  \item{outputBias}{Option whether to use the bias parameter in the output layer}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{The best number of hidden neurons found by the automatic procedure.}
\references{
  Lima, A.R., A.J. Cannon and W.W. Hsieh. Nonlinear regression in environmental sciences using extreme learning machines.   Environmental Modelling and Software (submitted 2014/2/3) 
  
  Yuret, D., 1994. From genetic algorithms to efficient optimization. 
  Technical Report 1569. MIT AI Laboratory.
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
set.seed(123)
attach(wtloss)
library("scales")

#scaling the inputs/outputs
x.train <- rescale(as.matrix(wtloss$Days), to=c(-1,1))
y.train <- rescale(as.matrix(wtloss$Weight), to=c(-1,1))

#Finding the best number of hidden neurons
number.hn <- Elm.search.hc(x.train,y.train)

#training the ELM
trained.elm <- Elm.train(x.train,y.train,Number.hn = number.hn)

#rescaling back the elm outputs
elm.fit.values <- rescale(trained.elm$predictionTrain,to= range(as.matrix(wtloss$Weight)),from=c(-1,1))

oldpar <- par(mar = c(5.1, 4.1, 4.1, 4.1))
plot(wtloss$Days, wtloss$Weight, type = "p", ylab = "Weight (kg)",main="Weight Reduction")
lines(wtloss$Days, elm.fit.values,col=2,type='b')
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line

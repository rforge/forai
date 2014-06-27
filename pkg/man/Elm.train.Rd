\name{Elm.train}
\alias{Elm.train}
  
\title{Fit Extreme Learning Machine (ELM)}
\description{
  Fit Extreme Learning Machine.
}
\usage{
Elm.train(X.fit, Y.fit, Number.hn = 10, autorangeweight = FALSE, 
          rangeweight = NULL, activation = "TANH", outputBias = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X.fit}{Data matrix (numeric) containing the input values (predictors) used to train the model.}
  \item{Y.fit}{Response vector (numeric) used to train the model.}
  \item{Number.hn}{Number of units in the hidden layer. Default is \code{10}.}
  \item{autorangeweight}{Option whether to use the automated range used for the weights. Default is \code{FALSE}.}
  \item{rangeweight}{Initial random weights on \code{[-rangeweight,rangeweight]}. The default is \code{NULL}}
%  \item{PenaltyP}{Values of the regularization factor. Default is \code{0}.}
  \item{activation}{Activation function of the hidden layer neurons. Available functions are: 'TANH' and 'SIG'.}
  \item{outputBias}{Option whether to use the bias parameter in the output layer}
}

\value{
  \item{inputWeight}{Set of weights used.}
  \item{biasofHN}{Set of bias used.}
  \item{matrixBeta}{{Set of weights adjusted.}
  \item{predictionTrain}{The fitted values for the training data.}
  \item{rangeweight}{Used range of the random weight initialization (\code{[-rangeweight,rangeweight]}},
  \item{activation}{Activation function of the hidden layer neurons.}
  \item{outputBias}{Option whether to use the bias parameter in the output layer.}
}
\references{
       G.-B. Huang, Q.-Y. Zhu, C.-K. Siew (2006)                              
       Extreme learning machine: Theory and applications                      
       Neurocomputing 70 (2006) 489-501                                       
}

\author{Aranildo Lima}

\note{To achieve better results, the use of a pre-processing technique (e.g. standardization of variables) is
important.}

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

  #training the ELM
  trained.elm <- Elm.train(x.train,y.train,Number.hn =5)
  
  #rescaling back the elm outputs
  elm.fit.values <- rescale(trained.elm$predictionTrain,to= range(as.matrix(wtloss$Weight)))
  
  RMSE(wtloss$Weight,elm.fit.values)
  
  oldpar <- par(mar = c(5.1, 4.1, 4.1, 4.1))
  plot(wtloss$Days, wtloss$Weight, type = "p", ylab = "Weight (kg)",main="Weight Reduction")
  lines(wtloss$Days, elm.fit.values,col=2,type='b')
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line

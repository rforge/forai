\name{Elm.predict}
\alias{Elm.predict}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{ELM Predictions}
\description{Function for predictions from the results of ELM model fitting function.}
\usage{
Elm.predict(TrainedElm, X.fit)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{TrainedElm}{an ELM object for which prediction is desired.}
  \item{X.fit}{Data matrix (numeric) containing the input values (predictors).}
}

\value{The fitted values.}

\references{
       G.-B. Huang, Q.-Y. Zhu, C.-K. Siew (2006)                              
       Extreme learning machine: Theory and applications                      
       Neurocomputing 70 (2006) 489-501                                       
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
  
  train.index <- sample(length(wtloss$Days),size=40)
  
  #scaling the inputs/outputs
  x.train <- rescale(as.matrix(wtloss$Days), to=c(-1,1))[train.index]
  y.train <- rescale(as.matrix(wtloss$Weight), to=c(-1,1))[train.index]
  x.test <- rescale(as.matrix(wtloss$Days), to=c(-1,1))[-train.index]
  
  #training the ELM
  trained.elm <- Elm.train(x.train,y.train,Number.hn =5)
  
  #rescaling back the elm outputs
  elm.fit.values <- rescale(trained.elm$predictionTrain,
                            to= range(as.matrix(wtloss$Weight)),from=c(-1,1))
  elm.predicted.values <- rescale(Elm.predict(trained.elm, x.test),
                                to= range(as.matrix(wtloss$Weight)),from=c(-1,1))
  
  oldpar <- par(mar = c(5.1, 4.1, 4.1, 4.1))
  plot(wtloss$Days, wtloss$Weight, type = "p", ylab = "Weight (kg)",main="Weight Reduction",pch=20)
  points(wtloss$Days[train.index], elm.fit.values,col=2,type='p',pch=20)
  points(wtloss$Days[-train.index], elm.predicted.values,col=4,type='p',pch=20)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line

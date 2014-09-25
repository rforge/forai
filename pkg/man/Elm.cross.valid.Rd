\name{Elm.cross.valid}
\alias{Elm.cross.valid}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Training ELM with cross validation.}
\description{A k-fold cross validation on the training data is performed.}
\usage{
Elm.cross.valid(X.fit, Y.fit, Number.hn=10, n.blocks=5, returnmodels = FALSE, 
                            autorangeweight=FALSE, rangeweight=1, 
                            activation='TANH',outputBias = FALSE,
                            rangebias = 1)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X.fit}{Data matrix (numeric) containing the input values (predictors) used to train the model.}
  \item{Y.fit}{Response vector (numeric) used to train the model.}
  \item{Number.hn}{Number of units in the hidden layer. Default is \code{10}.}
  \item{n.blocks}{an integer specifying the desired number of cross-validation folds. Default is \code{5}.}
  \item{returnmodels}{whether to return the trained models. Default is \code{FALSE}.}
  \item{autorangeweight}{Option whether to use the automated range used for the weights. Default is \code{FALSE}.}
  \item{rangeweight}{Initial random weights on \code{[-rangeweight,rangeweight]}. The default is \code{1}}
%  \item{PenaltyP}{Values of the regularization factor. Default is \code{0}.}
  \item{activation}{Activation function of the hidden layer neurons. Available functions are: 'TANH' (default) and 'SIG'.}
  \item{outputBias}{Option whether to use the bias parameter in the output layer}
  \item{rangebias}{Initial random bias on \code{[-rangebias,rangebias]}. The default is \code{1}}
}

\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
If \code{returnmodels=FALSE} (default), it returns the fitted values. If \code{returnmodels=TRUE}, it returns a list containing:.
  \item{predictionTrain}{The fitted values for the training data.}
  \item{trained.elms}{A list containing the k models used.}
}
\references{
%% ~put references to the literature/web site here ~
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

#training the ELM
elm.fit.values <- rescale(Elm.cross.valid(x.train,y.train,Number.hn=5,n.blocks=5),
                          to= range(as.matrix(wtloss$Weight)),from=c(-1,1))

oldpar <- par(mar = c(5.1, 4.1, 4.1, 4.1))
plot(wtloss$Days, wtloss$Weight, type = "p", ylab = "Weight (kg)",
      main="Weight Reduction",pch=20)
points(wtloss$Days, elm.fit.values,col=2,type='p',pch=20)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line

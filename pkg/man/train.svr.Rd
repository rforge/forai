\name{train.svr}
\alias{train.svr}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
train.svr
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
train.svr(xtrain, ytrain, hypparameter, ErrorFunc, PercentValid = 20, kfold = 1, SplitRandom = FALSE, kernel = "radial")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{xtrain}{
%%     ~~Describe \code{xtrain} here~~
}
  \item{ytrain}{
%%     ~~Describe \code{ytrain} here~~
}
  \item{hypparameter}{
%%     ~~Describe \code{hypparameter} here~~
}
  \item{ErrorFunc}{
%%     ~~Describe \code{ErrorFunc} here~~
}
  \item{PercentValid}{
%%     ~~Describe \code{PercentValid} here~~
}
  \item{kfold}{
%%     ~~Describe \code{kfold} here~~
}
  \item{SplitRandom}{
%%     ~~Describe \code{SplitRandom} here~~
}
  \item{kernel}{
%%     ~~Describe \code{kernel} here~~
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
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
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (xtrain, ytrain, hypparameter, ErrorFunc, PercentValid = 20, 
    kfold = 1, SplitRandom = FALSE, kernel = "radial") 
{
    if (kfold > 1) {
        n.cases = nrow(xtrain)
        index.block <- xval.buffer(n.cases, kfold)
        pred.valid <- rep(Inf, n.cases)
        for (nb in 1:kfold) {
            svr.try <- try(svm(xtrain[index.block[[nb]]$train, 
                , drop = FALSE], ytrain[index.block[[nb]]$train, 
                , drop = FALSE], kernel = kernel, gamma = hypparameter[1], 
                epsilon = hypparameter[2], cost = hypparameter[3]), 
                silent = TRUE)
            if (class(svr.try) != "try-error") {
                pred.valid[index.block[[nb]]$valid] = predict(svr.try, 
                  xtrain[index.block[[nb]]$valid, , drop = FALSE])
            }
            else {
                return(list(error.svm = TRUE))
            }
        }
        return(list(error.svm = FALSE, fitted = pred.valid, ffValid = ErrorFunc(ytrain, 
            pred.valid)))
    }
    else {
        nTrain = nrow(xtrain)
        indValid <- nTrain - round((nTrain * (PercentValid/100)))
        if (SplitRandom) {
            cases <- sample(nTrain)
            x.fit <- x.train[cases, , drop = FALSE]
            y.fit <- y.train[cases]
        }
        x.fit.train <- xtrain[1:indValid, , drop = FALSE]
        x.fit.valid <- xtrain[(indValid + 1):nTrain, , drop = FALSE]
        y.fit.train <- ytrain[1:indValid]
        y.fit.valid <- ytrain[(indValid + 1):nTrain]
        svr.try <- try(svm(x.fit.train, y.fit.train, kernel = "radial", 
            gamma = hypparameter[1], epsilon = hypparameter[2], 
            cost = hypparameter[3]), silent = TRUE)
        if (class(svr.try) != "try-error") {
            sv <- svr.try
            return(list(error.svm = FALSE, fitted.train = sv$fitted, 
                fitted.valid = predict(sv, x.fit.valid), ffValid = ErrorFunc(y.fit.valid, 
                  predict(sv, x.fit.valid)), ffTrain = ErrorFunc(y.fit.train, 
                  sv$fitted)))
        }
        else {
            return(list(error.svm = TRUE))
        }
    }
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line

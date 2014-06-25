\name{Elm.search.hn}
\alias{Elm.search.hn}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
Elm.search.hn(X.fit, Y.fit, n.ensem = 10, n.blocks = 5, ErrorFunc = RMSE, percentValid = 20, maxHiddenNodes = NULL, Trace = TRUE, autorangeweight = FALSE, outputBias = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X.fit}{
%%     ~~Describe \code{X.fit} here~~
}
  \item{Y.fit}{
%%     ~~Describe \code{Y.fit} here~~
}
  \item{n.ensem}{
%%     ~~Describe \code{n.ensem} here~~
}
  \item{n.blocks}{
%%     ~~Describe \code{n.blocks} here~~
}
  \item{ErrorFunc}{
%%     ~~Describe \code{ErrorFunc} here~~
}
  \item{percentValid}{
%%     ~~Describe \code{percentValid} here~~
}
  \item{maxHiddenNodes}{
%%     ~~Describe \code{maxHiddenNodes} here~~
}
  \item{Trace}{
%%     ~~Describe \code{Trace} here~~
}
  \item{autorangeweight}{
%%     ~~Describe \code{autorangeweight} here~~
}
  \item{outputBias}{
%%     ~~Describe \code{outputBias} here~~
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
function (X.fit, Y.fit, n.ensem = 10, n.blocks = 5, ErrorFunc = RMSE, 
    percentValid = 20, maxHiddenNodes = NULL, Trace = TRUE, autorangeweight = FALSE, 
    outputBias = FALSE) 
{
    acceleration <- 1.51
    candidates <- c((-1/4 * acceleration), 0, (1/acceleration), 
        acceleration)
    currentPoint <- 3
    stepSize <- 2
    cand.error.train <- rep(Inf, 4)
    cand.error.valid <- rep(Inf, 4)
    bestHN <- c(Inf, Inf, Inf)
    n.cases = length(Y.fit)
    if (n.blocks != 1) {
        index.block <- xval.buffer(n.cases, n.blocks)
    }
    else {
        indValid <- n.cases - round((n.cases * (percentValid/100)))
    }
    if (is.null(maxHiddenNodes)) {
        maxHiddenNodes <- (n.cases - 1)
    }
    TOP <- TRUE
    while (TOP) {
        if (n.blocks != 1) {
            pred.ens.valid <- matrix(NA, n.cases, n.ensem)
        }
        else {
            pred.ens.train <- matrix(NA, indValid, n.ensem)
            pred.ens.valid <- matrix(NA, (n.cases - indValid), 
                n.ensem)
        }
        cand.error.train[2] = bestHN[2]
        cand.error.valid[2] = bestHN[3]
        for (ic in c(1, 3, 4)) {
            for (e in 1:n.ensem) {
                n.hidden.can <- round(currentPoint + stepSize * 
                  candidates[ic])
                if (n.blocks != 1) {
                  pred.ens.valid[, e] = Elm.cross.valid(X.fit, 
                    Y.fit, n.hidden.can, n.blocks, autorangeweight, 
                    outputBias = outputBias)
                }
                else {
                  fit.elm <- Elm.optmization(X.fit[(1:indValid), 
                    , drop = FALSE], Y.fit[(1:indValid), drop = FALSE], 
                    Number.hn = n.hidden.can, autorangeweight = autorangeweight, 
                    outputBias = outputBias)
                  pred.ens.train[, e] = fit.elm$predictionTrain
                  pred.ens.valid[, e] = Elm.predict(fit.elm, 
                    X.fit[((indValid + 1):nTrain), , drop = FALSE])
                }
            }
            if (n.blocks != 1) {
                cand.error.train[ic] = cand.error.valid[ic] = ErrorFunc(Y.fit, 
                  rowMeans(pred.ens.valid))
            }
            else {
                cand.error.train[ic] = ErrorFunc(Y.fit[(1:indValid), 
                  drop = FALSE], rowMeans(pred.ens.train))
                cand.error.valid[ic] = ErrorFunc(Y.fit[((indValid + 
                  1):nTrain), drop = FALSE], rowMeans(pred.ens.valid))
            }
        }
        bestSolution <- which.min(cand.error.valid)
        if ((bestHN[3] > cand.error.valid[bestSolution]) & (bestHN[2] > 
            cand.error.train[bestSolution])) {
            if (bestHN[1] == max(1, round(currentPoint + stepSize * 
                candidates[bestSolution]))) {
                cat("REPITIU??: \n")
                break
            }
            bestHN[1] = max(1, round(currentPoint + stepSize * 
                candidates[bestSolution]))
            bestHN[2] = cand.error.train[bestSolution]
            bestHN[3] = cand.error.valid[bestSolution]
            if (Trace) {
                cat("hn: ", bestHN[1], " step:", stepSize, "RMSE Train:", 
                  cand.error.train[bestSolution], "RMSE Valid:", 
                  cand.error.valid[bestSolution], " bs:", bestSolution, 
                  " Cand:", round(currentPoint + stepSize * candidates), 
                  "\n")
            }
            if (bestSolution != 2) {
                currentPoint = bestHN[1]
                stepSize <- max(1, round(stepSize * candidates[bestSolution]))
            }
            if (bestHN[1] >= maxHiddenNodes) 
                break
        }
        else {
            break
        }
    }
    if (Trace) {
        cat("Best Solution: ", bestHN[1], "\n")
    }
    return(bestHN[1])
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line

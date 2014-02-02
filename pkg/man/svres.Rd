\name{svres}
\alias{svres}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
SVR-ES
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
svres(X.train, Y.train, X.test, PercentValid = 20, Generations = 500, InitialGamma = 0.001, ErrorFunc = MSE, Step = TRUE, StepBoruta = FALSE, SplitRandom = FALSE, Trace = TRUE, dTrace = 10, earlyStop = 50, kfold = 1)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X.train}{
%%     ~~Describe \code{X.train} here~~
}
  \item{Y.train}{
%%     ~~Describe \code{Y.train} here~~
}
  \item{X.test}{
%%     ~~Describe \code{X.test} here~~
}
  \item{PercentValid}{
%%     ~~Describe \code{PercentValid} here~~
}
  \item{Generations}{
%%     ~~Describe \code{Generations} here~~
}
  \item{InitialGamma}{
%%     ~~Describe \code{InitialGamma} here~~
}
  \item{ErrorFunc}{
%%     ~~Describe \code{ErrorFunc} here~~
}
  \item{Step}{
%%     ~~Describe \code{Step} here~~
}
  \item{StepBoruta}{
%%     ~~Describe \code{StepBoruta} here~~
}
  \item{SplitRandom}{
%%     ~~Describe \code{SplitRandom} here~~
}
  \item{Trace}{
%%     ~~Describe \code{Trace} here~~
}
  \item{dTrace}{
%%     ~~Describe \code{dTrace} here~~
}
  \item{earlyStop}{
%%     ~~Describe \code{earlyStop} here~~
}
  \item{kfold}{
%%     ~~Describe \code{kfold} here~~
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
function (X.train, Y.train, X.test, PercentValid = 20, Generations = 500, 
    InitialGamma = 0.001, ErrorFunc = MSE, Step = TRUE, StepBoruta = FALSE, 
    SplitRandom = FALSE, Trace = TRUE, dTrace = 10, earlyStop = 50, 
    kfold = 1) 
{
    if (Step & !StepBoruta) {
        targetTrain <- as.matrix(Y.train)
        colnames(targetTrain) <- c("Y")
        Xtarget <- cbind(X.train, targetTrain)
        fit.step <- lm(Y ~ ., as.data.frame(Xtarget))
        para.step <- step(fit.step, trace = 0)
        step.coef <- attr(para.step$terms, "term.labels")
        if (length(step.coef) == 0) {
            step.coef <- colnames(Xtarget)[which.max(abs(fit.step$coef[-1]))]
        }
        x.fit <- X.train[, step.coef, drop = FALSE]
        x.test <- X.test[, step.coef, drop = FALSE]
        stepvars <- colnames(X.train) \%in\% step.coef
    }
    else if (Step & StepBoruta) {
        trf <- suppressWarnings(tuneRF(X.train, Y.train, doBest = TRUE, 
            plot = FALSE, ntree = 500, nodesize = 1))
        bor <- Boruta(x = as.data.frame(X.train), y = Y.train, 
            light = FALSE, doTrace = 2, ntree = trf$ntree, mtry = trf$mtry, 
            nodesize = 1)
        bor <- suppressWarnings(TentativeRoughFix(bor))
        stepvars <- abs(as.integer(bor$finalDecision) - 3)
        stepvars <- stepvars == 1
        x.fit <- X.train[, stepvars, drop = FALSE]
        x.test <- X.test[, stepvars, drop = FALSE]
    }
    else {
        x.fit <- X.train
        x.test <- X.test
        stepvars <- rep(TRUE, ncol(X.train))
    }
    y.fit <- Y.train
    nTrain <- length(y.fit)
    nNeighbor <- 4
    kksvr <- knn.reg(x.fit, test = NULL, y = y.fit, k = nNeighbor)
    factor1 <- ((nTrain)^(1/5)) * nNeighbor
    factor2 <- ((nTrain)^(1/5)) * (nNeighbor - 1)
    errorEst <- sqrt((factor1/factor2) * (kksvr$PRESS/nTrain))
    epsilCal <- 3 * errorEst * (sqrt(log(nTrain)/nTrain))
    C <- mean(y.fit) + 3 * sd(c(y.fit))
    hypparameter <- c(InitialGamma, epsilCal, C)
    names(hypparameter) <- c("Gamma", "Epsilon", "C")
    fitted.svr <- train.svr(x.fit, y.fit, hypparameter, ErrorFunc, 
        PercentValid, kfold)
    ffValid <- fitted.svr$ffValid
    if (kfold == 1) {
        ffTrain <- fitted.svr$ffTrain
    }
    mutation <- hypparameter
    noevolution <- 0
    for (d in 1:Generations) {
        noevolution = noevolution + 1
        firstNorm <- rnorm(3)
        son <- abs(hypparameter + (mutation * firstNorm))
        tau <- 1/sqrt(2 * sqrt(3))
        taul <- 1/sqrt(6)
        firstNorm <- rnorm(1)
        secNorm <- rnorm(3)
        mutationtemp <- mutation * exp(taul * firstNorm + tau * 
            secNorm)
        fitted.svr <- train.svr(x.fit, y.fit, son, ErrorFunc, 
            PercentValid, kfold)
        if (fitted.svr$error.svm) {
            ffSonValid <- Inf
        }
        else {
            ffSonValid <- fitted.svr$ffValid
        }
        if (ffSonValid < ffValid) {
            if (kfold > 1) {
                hypparameter <- son
                mutation <- mutationtemp
                ffValid <- ffSonValid
                noevolution <- 0
            }
            else {
                ffSonTrain <- fitted.svr$ffTrain
                if (ffSonTrain < ffTrain) {
                  hypparameter <- son
                  mutation <- mutationtemp
                  ffValid <- ffSonValid
                  ffTrain <- ffSonTrain
                  noevolution <- 0
                }
            }
        }
        if (Trace & ((d\%\%dTrace) == 0)) {
            cat("Generation --> ", d, "\n")
            if (kfold > 1) {
                cat("Error: Valid --> ", ffValid, "\n")
            }
            else {
                cat("Error: Train -->", ffTrain, " | Valid --> ", 
                  ffValid, "\n")
            }
        }
        if (noevolution > earlyStop) {
            break
        }
    }
    sv <- svm(x.fit, y.fit, kernel = "radial", gamma = hypparameter[1], 
        epsilon = hypparameter[2], cost = hypparameter[3])
    p.svmES <- predict(sv, x.test)
    if (kfold > 1) {
        svres.results <- list(hypparameter = hypparameter, forecast = p.svmES, 
            svmf = sv, ffTrain = NULL, ffValid = ffValid, stepvars = stepvars)
    }
    else {
        svres.results <- list(hypparameter = hypparameter, forecast = p.svmES, 
            svmf = sv, ffTrain = ffTrain, ffValid = ffValid, 
            stepvars = stepvars)
    }
    return(svres.results)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line

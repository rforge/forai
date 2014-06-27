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
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (TrainedElm, X.fit) 
{
    X.fit <- t(X.fit)
    NumberofData = ncol(X.fit)
    B <- TrainedElm$matrixBeta
    inputWeight <- TrainedElm$inputWeight
    biasofHN <- TrainedElm$biasofHN
    tempH = inputWeight \%*\% X.fit
    ind = matrix(1, 1, NumberofData)
    BiasMatrix = biasofHN \%*\% ind
    rm(ind)
    tempH = tempH + BiasMatrix
    rm(BiasMatrix)
    switch(TrainedElm$activation, TANH = {
        H = t(tanh(tempH))
    }, {
        H = t(1/(1 + exp(-tempH)))
    })
    rm(tempH)
    if (TrainedElm$outputBias) {
        return(as.vector(unlist(t(cbind(1, H) \%*\% B))))
    }
    else {
        return(as.vector(unlist(t(H \%*\% B))))
    }
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line

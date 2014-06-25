\name{Elm.predict}
\alias{Elm.predict}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
Elm.predict(TrainedElm, X.fit)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{TrainedElm}{
%%     ~~Describe \code{TrainedElm} here~~
}
  \item{X.fit}{
%%     ~~Describe \code{X.fit} here~~
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

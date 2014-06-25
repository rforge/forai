\name{Elm.cross.valid}
\alias{Elm.cross.valid}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
Elm.cross.valid(X.fit, Y.fit, Number.hn, n.blocks = 5, autorangeweight = FALSE, outputBias = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X.fit}{
%%     ~~Describe \code{X.fit} here~~
}
  \item{Y.fit}{
%%     ~~Describe \code{Y.fit} here~~
}
  \item{Number.hn}{
%%     ~~Describe \code{Number.hn} here~~
}
  \item{n.blocks}{
%%     ~~Describe \code{n.blocks} here~~
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
function (X.fit, Y.fit, Number.hn, n.blocks = 5, autorangeweight = FALSE, 
    outputBias = FALSE) 
{
    n.cases = length(Y.fit)
    index.block <- xval.buffer(n.cases, n.blocks)
    pred.ens.valid <- matrix(NA, n.cases, 1)
    for (nb in 1:n.blocks) {
        fit.elm <- Elm.optmization(X.fit[index.block[[nb]]$train, 
            , drop = FALSE], Y.fit[index.block[[nb]]$train, , 
            drop = FALSE], Number.hn = Number.hn, autorangeweight = autorangeweight, 
            outputBias = outputBias)
        pred.ens.valid[index.block[[nb]]$valid, 1] = Elm.predict(fit.elm, 
            X.fit[index.block[[nb]]$valid, , drop = FALSE])
    }
    return(pred.ens.valid)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line

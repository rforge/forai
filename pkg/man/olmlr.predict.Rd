\name{olmlr.predict}
\alias{olmlr.predict}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{ELM Predictions}
\description{Function for predictions from the results of OL-MLR model fitting function.}
\usage{
olmlr.predict(TrainedMLR, X.fit)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{TrainedMLR}{a MLR object for which prediction is desired.}
  \item{X.fit}{Data matrix (numeric) containing the input values (predictors).}
}

\value{The fitted values.}

\references{
  Ross, G. H., 1987: An updateable model output statistics scheme.
  Programme on Short- and Medium Range Weather Prediction,
  PSMP Report Series, No. 25, World Meteorological Organization, 25–28.

  Laurence J. Wilson and Marcel Vallee, 2002: The Canadian Updateable Model Output Statistics (UMOS) System:          
  Design and Development Tests. Wea. Forecasting, 17, 206–222
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
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line

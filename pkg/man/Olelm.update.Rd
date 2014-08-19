\name{Olelm.update}
\alias{Olelm.update}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Update the online ELM}
\description{Update the online ELM}
\usage{
Olelm.update(TrainedElm, ppredictors, ptarget)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{TrainedElm}{an ELM object.}
  \item{ppredictors}{Data matrix (numeric) containing the input values (predictors) used to train the model.}
  \item{ptarget}{Response vector (numeric) used to train the model.}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
  \item{inputWeight}{Set of weights used.}
  \item{biasofHN}{Set of bias used.}
  \item{matrixBeta}{Set of weights adjusted.}
  \item{matrixP}{matrixP.}
  \item{predictionTrain}{The fitted values for the training data.}
  \item{rangeweight}{Used range of the random weight initialization (\code{[-rangeweight,rangeweight]}},
  \item{activation}{Activation function of the hidden layer neurons.}
  \item{outputBias}{Option whether to use the bias parameter in the output layer.}
}
\references{
%% ~put references to the literature/web site here ~
}
\author{Aranildo Lima}

\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line

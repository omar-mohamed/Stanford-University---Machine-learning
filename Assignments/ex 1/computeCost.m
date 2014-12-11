function J = computeCost(X, y, theta)
m = length(y); 
predictions=X*theta;
errors=predictions-y;
sumSqrErrors=errors'*errors;
J=1/(2*m)* sumSqrErrors;


end

meD=[10 20 30 40 50]; %mean for ill – five tests
deD=[1 2 3 4 5]; %variance for ill – five tests
meH=[5 10 20 30 40]; %mean for healthy – five tests
deH=[1 2 3 4 5]; %variance for healthy– five tests

samples = 1000;

healthy = zeros(samples,5);
ill = zeros(samples,5);

healthy =  readmatrix('healthy.csv'); 
ill =  readmatrix('ill.csv');
% task 1
for i=1:5
%     healthy(:,i) =  deH(i).*randn(samples,1) + meH(i); %%% CREATA DATA
%     ill(:,i) =  deD(i).*randn(samples,1) + meD(i);
    figure
    histogram(healthy(:,i))
    hold on
    histogram(ill(:,i))
    legend('Healthy','Ill')
    hold off
end
% csvwrite('healthy.csv', healthy)
% csvwrite('ill.csv', ill)

% task 2
H = zeros(1,5);
pValueHealthy = zeros(1,5);
pValueIll = zeros(1,5);
W = zeros(1,5);
for i=1:5
    [H(1,i), pValueHealthy(1,i), W(1,i)] = swtest(healthy(:,i), 0.05);
    [H(1,i), pValueIll(1,i), W(1,i)] = swtest(ill(:,i), 0.05);
    pValueHealthy(1,i)
    pValueIll(1,i)
    figure
    qqplot(healthy(:,i))
    figure
    qqplot(ill(:,i))
end

% task 3 ROC Curves
crossp = zeros(1,5);
for i=1:5
    crossp(i) = crosspoint(meD(i), meH(i), deD(i), deH(i));
end

kosz = 100;

ile_ill = zeros(5,kosz);
ile_healthy = zeros(5,kosz);
for k=1:5
    min_ill = min(ill(:,k));
    max_ill = max(ill(:,k));
    diff = (max_ill - min_ill)/kosz;
    low = min_ill;
    high = min_ill + diff;
    for i=1:kosz
        for j=1:samples
            if low <= ill(j,k) && high > ill(j,k)
                ile_ill(k,i) = ile_ill(k,i) + 1;
            end
            if low <= healthy(j,k) && high > healthy(j,k)
                ile_healthy(k,i) = ile_healthy(k,i) + 1;
            end
        end
        low = low + diff;
        high = high + diff;
    end
end

for k=1:5
    for i=1:kosz
        pass(k,i) = sum(ile_ill(k,(1:i)));
    end
    if pass(k,kosz) == 999
        pass(k,kosz) = pass(k,kosz) + 1;
    end
end
for k=1:5
    for i=1:kosz
        fail(k,i) = sum(ile_healthy(k,(1:i)));
    end
end

for k=1:5
    for i=1:kosz
        FPR(k,i) = 1 - (pass(k,i)/pass(k,kosz));
        TRP(k,i) = 1 - (fail(k,i)/fail(k,kosz));
    end
    figure
    plot(TRP(k,:),FPR(k,:))
    title('ROC Curve', num2str(k))
    xlabel('TPR')
    ylabel('FPR')
end
for k=1:10
%     x = zeros(samples,6);    %%%% CREATING 10 TESTS 
%     for i=1:1000
%         for j=1:5
%             if i <= 500
%                x(i,j) = deH(j).*randn(1,1) + meH(j);
%             else
%                x(i,j) = deD(j).*randn(1,1) + meD(j);
%                x(i,6) = 1;
%             end
%         end
%     end
%    writematrix(x, num2str(k,'test%02d.csv'));
    x = readmatrix(num2str(k,'test%02d.csv'));
    test_results = zeros(1000,31);
    for i=1:1000
        [test_results(i,:), TP(i,:), FP(i,:)] = rules(x(i,1:6), crossp);
    end
    for i=1:31
        sum_TP(k,i) = sum(TP(:,i));
        sum_FP(k,i) = sum(FP(:,i));
    end
end

for i=1:10
    figure
    bar(sum_TP(i,:),'red')
    title('True Positive', num2str(i))
    figure
    bar(sum_FP(i,:))
    title("False Positive", num2str(i))
end


for i=1:31
    figure
    h = plot(sum_FP(:,i),sum_TP(:,i),'*');
    set(h,'linewidth',4);
    xt = xticks;
    if max(xt) == 1 && min(xt) == 0
        xlim([-0.2 1.2])
    end
    title('rule', num2str(i))
    xlabel('FP')
    ylabel('TP')
end


function [crosspoint] = crosspoint(meanIll, meanHeal, STDIll, STDHeal)
  a = (1/(STDHeal.^2)) - (1/(STDIll.^2));
  b = 2*((meanIll/(STDIll.^2)) - (meanHeal/(STDHeal.^2)));
  c = ((meanHeal.^2)/(STDHeal.^2)) - ((meanIll.^2)/(STDIll.^2)) + log((STDHeal.^2)/(STDIll.^2));
  x = [a b c];
  crosspoint = roots(x);
end

function [x, TP, FP] = rules(p, crosspoint)
x = zeros(1,31);
TP = zeros(1,31);
FP = zeros(1,31);
if p(1) > crosspoint(1)
   x(1,1) = 1;
   if p(6) == 1
       TP(1) = TP(1) + 1;
   else
       FP(1) = FP(1) + 1;
   end
end
if p(2) > crosspoint(2)
   x(1,2) = 1;
   if p(6) == 1
       TP(2) = TP(2) + 1;
   else
       FP(2) = FP(2) + 1;
   end
end
if p(3) > crosspoint(3)
   x(1,3) = 1;
   if p(6) == 1
       TP(3) = TP(3) + 1;
   else
       FP(3) = FP(3) + 1;
   end
end
if p(4) > crosspoint(4)
   x(1,4) = 1;
   if p(6) == 1
       TP(4) = TP(4) + 1;
   else
       FP(4) = FP(4) + 1;
   end
end
if p(5) > crosspoint(5)
   x(1,5) = 1;
   if p(6) == 1
       TP(5) = TP(5) + 1;
   else
       FP(5) = FP(5) + 1;
   end
end
if p(1) > crosspoint(1) && p(2) > crosspoint(2)
   x(1,6) = 1;
   if p(6) == 1
       TP(6) = TP(6) + 1;
   else
       FP(6) = FP(6) + 1;
   end
end
if p(1) > crosspoint(1) && p(3) > crosspoint(3)
   x(1,7) = 1;
   if p(6) == 1
       TP(7) = TP(7) + 1;
   else
       FP(7) = FP(7) + 1;
   end
end
if p(1) > crosspoint(1) && p(4) > crosspoint(4)
   x(1,8) = 1;
   if p(6) == 1
       TP(8) = TP(8) + 1;
   else
       FP(8) = FP(8) + 1;
   end
end
if p(1) > crosspoint(1) && p(5) > crosspoint(5)
   x(1,9) = 1;
   if p(6) == 1
       TP(9) = TP(9) + 1;
   else
       FP(9) = FP(9) + 1;
   end
end
if p(2) > crosspoint(2) && p(3) > crosspoint(3)
   x(1,10) = 1;
   if p(6) == 1
       TP(10) = TP(10) + 1;
   else
       FP(10) = FP(10) + 1;
   end
end
if p(2) > crosspoint(2) && p(4) > crosspoint(4)
   x(1,11) = 1;
   if p(6) == 1
       TP(11) = TP(11) + 1;
   else
       FP(11) = FP(11) + 1;
   end
end
if p(2) > crosspoint(2) && p(5) > crosspoint(5)
   x(1,12) = 1;
   if p(6) == 1
       TP(12) = TP(12) + 1;
   else
       FP(12) = FP(12) + 1;
   end
end
if p(3) > crosspoint(3) && p(4) > crosspoint(4)
   x(1,13) = 1;
   if p(6) == 1
       TP(13) = TP(13) + 1;
   else
       FP(13) = FP(13) + 1;
   end
end
if p(3) > crosspoint(3) && p(5) > crosspoint(5)
   x(1,14) = 1;
   if p(6) == 1
       TP(14) = TP(14) + 1;
   else
       FP(14) = FP(14) + 1;
   end
end
if p(4) > crosspoint(4) && p(5) > crosspoint(5)
   x(1,15) = 1;
   if p(6) == 1
       TP(15) = TP(15) + 1;
   else
       FP(15) = FP(15) + 1;
   end
end
if p(1) > crosspoint(1) && p(2) > crosspoint(2) && p(3) > crosspoint(3)
   x(1,16) = 1;
   if p(6) == 1
       TP(16) = TP(16) + 1;
   else
       FP(16) = FP(16) + 1;
   end
end
if p(1) > crosspoint(1) && p(2) > crosspoint(2) && p(4) > crosspoint(4)
   x(1,17) = 1;
   if p(6) == 1
       TP(17) = TP(17) + 1;
   else
       FP(17) = FP(17) + 1;
   end
end
if p(1) > crosspoint(1) && p(2) > crosspoint(2) && p(5) > crosspoint(5)
   x(1,18) = 1;
   if p(6) == 1
       TP(18) = TP(18) + 1;
   else
       FP(18) = FP(18) + 1;
   end
end
if p(1) > crosspoint(1) && p(3) > crosspoint(3) && p(4) > crosspoint(4)
   x(1,19) = 1;
   if p(6) == 1
       TP(19) = TP(19) + 1;
   else
       FP(19) = FP(19) + 1;
   end
end
if p(1) > crosspoint(1) && p(3) > crosspoint(3) && p(5) > crosspoint(5)
   x(1,20) = 1;
   if p(6) == 1
       TP(20) = TP(20) + 1;
   else
       FP(20) = FP(20) + 1;
   end
end
if p(1) > crosspoint(1) && p(4) > crosspoint(4) && p(5) > crosspoint(5)
   x(1,21) = 1;
   if p(6) == 1
       TP(21) = TP(21) + 1;
   else
       FP(21) = FP(21) + 1;
   end
end
if p(2) > crosspoint(2) && p(3) > crosspoint(3) && p(4) > crosspoint(4)
   x(1,22) = 1;
   if p(6) == 1
       TP(22) = TP(22) + 1;
   else
       FP(22) = FP(22) + 1;
   end
end
if p(2) > crosspoint(2) && p(3) > crosspoint(3) && p(5) > crosspoint(5)
   x(1,23) = 1;
   if p(6) == 1
       TP(23) = TP(23) + 1;
   else
       FP(23) = FP(23) + 1;
   end
end
if p(2) > crosspoint(2) && p(4) > crosspoint(4) && p(5) > crosspoint(5)
   x(1,24) = 1; 
   if p(6) == 1
       TP(24) = TP(24) + 1;
   else
       FP(24) = FP(24) + 1;
   end
end
if p(3) > crosspoint(3) && p(4) > crosspoint(4) && p(5) > crosspoint(5)
   x(1,25) = 1;
   if p(6) == 1
       TP(25) = TP(25) + 1;
   else
       FP(25) = FP(25) + 1;
   end
end
if p(1) > crosspoint(1) && p(2) > crosspoint(2) && p(3) > crosspoint(3) && p(4) > crosspoint(4)
   x(1,26) = 1; 
   if p(6) == 1
       TP(26) = TP(26) + 1;
   else
       FP(26) = FP(26) + 1;
   end
end
if p(1) > crosspoint(1) && p(2) > crosspoint(2) && p(3) > crosspoint(3) && p(5) > crosspoint(5)
   x(1,27) = 1;
   if p(6) == 1
       TP(27) = TP(27) + 1;
   else
       FP(27) = FP(27) + 1;
   end
end
if p(1) > crosspoint(1) && p(2) > crosspoint(2) && p(4) > crosspoint(4) && p(5) > crosspoint(5)
   x(1,28) = 1;
   if p(6) == 1
       TP(28) = TP(28) + 1;
   else
       FP(28) = FP(28) + 1;
   end
end
if p(1) > crosspoint(1) && p(3) > crosspoint(3) && p(4) > crosspoint(4) && p(5) > crosspoint(5)
   x(1,29) = 1;
   if p(6) == 1
       TP(29) = TP(29) + 1;
   else
       FP(29) = FP(29) + 1;
   end
end
if p(2) > crosspoint(2) && p(3) > crosspoint(3) && p(4) > crosspoint(4) && p(5) > crosspoint(5)
   x(1,30) = 1;
   if p(6) == 1
       TP(30) = TP(30) + 1;
   else
       FP(30) = FP(30) + 1;
   end
end
if p(1) > crosspoint(1) && p(2) > crosspoint(2) && p(3) > crosspoint(3) && p(4) > crosspoint(4) && p(5) > crosspoint(5)
   x(1,31) = 1;
   if p(6) == 1
       TP(31) = TP(31) + 1;
   else
       FP(31) = FP(31) + 1;
   end
end
end

function [TP_count, FP_count, TN_count, FN_count] = tp_fp(x, crossp)

clas = zeros(1,1000);
for i=1:1000
    if sum(x(i,6:10)) >= 3
        clas(i) = 1;
    else
        clas(i) = 0;
    end
end

test_results = zeros(1000,31);
for i=1:1000
    [test_results(i,:), TP, FP] = rules(x(i,1:6), crossp);
end

clas_result = zeros(1000,31);
for i=1:500 % zdrowi
    for j=1:31
        if test_results(i,j) == clas(i)
            clas_result(i,j) = 1; % 1 = TP
        else
            clas_result(i,j) = 0; % 0 = FP
        end
    end
end

for i=501:1000 % chorzy
    for j=1:31
        if test_results(i,j) == clas(i)
            clas_result(i,j) = 1; % 1 = TN
        else
            clas_result(i,j) = 0; % 0 = FN
        end
    end
end

TP_count=zeros(1,31);
for i=1:31
    TP_count(i) = sum(clas_result(:,i));
end
FP_count = 1000 - TP_count(:)';

end
function [H, pValue, W] = swtest(x, alpha)
%SWTEST Shapiro-Wilk parametric hypothesis test of composite normality.
%   [H, pValue, SWstatistic] = SWTEST(X, ALPHA) performs the
%   Shapiro-Wilk test to determine if the null hypothesis of
%   composite normality is a reasonable assumption regarding the
%   population distribution of a random sample X. The desired significance 
%   level, ALPHA, is an optional scalar input (default = 0.05).
%
%   The Shapiro-Wilk and Shapiro-Francia null hypothesis is: 
%   "X is normal with unspecified mean and variance."
%
%   This is an omnibus test, and is generally considered relatively
%   powerful against a variety of alternatives.
%   Shapiro-Wilk test is better than the Shapiro-Francia test for
%   Platykurtic sample. Conversely, Shapiro-Francia test is better than the
%   Shapiro-Wilk test for Leptokurtic samples.
%
%   When the series 'X' is Leptokurtic, SWTEST performs the Shapiro-Francia
%   test, else (series 'X' is Platykurtic) SWTEST performs the
%   Shapiro-Wilk test.
% 
%    [H, pValue, SWstatistic] = SWTEST(X, ALPHA)
%
% Inputs:
%   X - a vector of deviates from an unknown distribution. The observation
%     number must exceed 3 and less than 5000.
%
% Optional inputs:
%   ALPHA - The significance level for the test (default = 0.05).
%  
% Outputs:
%  SWstatistic - The test statistic (non normalized).
%
%   pValue - is the p-value, or the probability of observing the given
%     result by chance given that the null hypothesis is true. Small values
%     of pValue cast doubt on the validity of the null hypothesis.
%
%     H = 0 => Do not reject the null hypothesis at significance level ALPHA.
%     H = 1 => Reject the null hypothesis at significance level ALPHA.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Copyright (c) 17 March 2009 by Ahmed Ben Saďda          %
%                 Department of Finance, IHEC Sousse - Tunisia           %
%                       Email: ahmedbensaida@yahoo.com                   %
%                    $ Revision 3.0 $ Date: 18 Juin 2014 $               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%
% References:
%
% - Royston P. "Remark AS R94", Applied Statistics (1995), Vol. 44,
%   No. 4, pp. 547-551.
%   AS R94 -- calculates Shapiro-Wilk normality test and P-value
%   for sample sizes 3 <= n <= 5000. Handles censored or uncensored data.
%   Corrects AS 181, which was found to be inaccurate for n > 50.
%   Subroutine can be found at: http://lib.stat.cmu.edu/apstat/R94
%
% - Royston P. "A pocket-calculator algorithm for the Shapiro-Francia test
%   for non-normality: An application to medicine", Statistics in Medecine
%   (1993a), Vol. 12, pp. 181-184.
%
% - Royston P. "A Toolkit for Testing Non-Normality in Complete and
%   Censored Samples", Journal of the Royal Statistical Society Series D
%   (1993b), Vol. 42, No. 1, pp. 37-43.
%
% - Royston P. "Approximating the Shapiro-Wilk W-test for non-normality",
%   Statistics and Computing (1992), Vol. 2, pp. 117-119.
%
% - Royston P. "An Extension of Shapiro and Wilk's W Test for Normality
%   to Large Samples", Journal of the Royal Statistical Society Series C
%   (1982a), Vol. 31, No. 2, pp. 115-124.
%

%
% Ensure the sample data is a VECTOR.
%

if numel(x) == length(x)
    x  =  x(:);               % Ensure a column vector.
else
    error(' Input sample ''X'' must be a vector.');
end

%
% Remove missing observations indicated by NaN's and check sample size.
%

x  =  x(~isnan(x));

if length(x) < 3
   error(' Sample vector ''X'' must have at least 3 valid observations.');
end

if length(x) > 5000
    warning('Shapiro-Wilk test might be inaccurate due to large sample size ( > 5000).');
end

%
% Ensure the significance level, ALPHA, is a 
% scalar, and set default if necessary.
%

if (nargin >= 2) && ~isempty(alpha)
   if ~isscalar(alpha)
      error(' Significance level ''Alpha'' must be a scalar.');
   end
   if (alpha <= 0 || alpha >= 1)
      error(' Significance level ''Alpha'' must be between 0 and 1.'); 
   end
else
   alpha  =  0.05;
end

% First, calculate the a's for weights as a function of the m's
% See Royston (1992, p. 117) and Royston (1993b, p. 38) for details
% in the approximation.

x       =   sort(x); % Sort the vector X in ascending order.
n       =   length(x);
mtilde  =   norminv(((1:n)' - 3/8) / (n + 1/4));
weights =   zeros(n,1); % Preallocate the weights.

if kurtosis(x) > 3
    
    % The Shapiro-Francia test is better for leptokurtic samples.
    
    weights =   1/sqrt(mtilde'*mtilde) * mtilde;

    %
    % The Shapiro-Francia statistic W' is calculated to avoid excessive
    % rounding errors for W' close to 1 (a potential problem in very
    % large samples).
    %

    W   =   (weights' * x)^2 / ((x - mean(x))' * (x - mean(x)));

    % Royston (1993a, p. 183):
    nu      =   log(n);
    u1      =   log(nu) - nu;
    u2      =   log(nu) + 2/nu;
    mu      =   -1.2725 + (1.0521 * u1);
    sigma   =   1.0308 - (0.26758 * u2);

    newSFstatistic  =   log(1 - W);

    %
    % Compute the normalized Shapiro-Francia statistic and its p-value.
    %

    NormalSFstatistic =   (newSFstatistic - mu) / sigma;
    
    % Computes the p-value, Royston (1993a, p. 183).
    pValue   =   1 - normcdf(NormalSFstatistic, 0, 1);
    
else
    
    % The Shapiro-Wilk test is better for platykurtic samples.

    c    =   1/sqrt(mtilde'*mtilde) * mtilde;
    u    =   1/sqrt(n);

    % Royston (1992, p. 117) and Royston (1993b, p. 38):
    PolyCoef_1   =   [-2.706056 , 4.434685 , -2.071190 , -0.147981 , 0.221157 , c(n)];
    PolyCoef_2   =   [-3.582633 , 5.682633 , -1.752461 , -0.293762 , 0.042981 , c(n-1)];

    % Royston (1992, p. 118) and Royston (1993b, p. 40, Table 1)
    PolyCoef_3   =   [-0.0006714 , 0.0250540 , -0.39978 , 0.54400];
    PolyCoef_4   =   [-0.0020322 , 0.0627670 , -0.77857 , 1.38220];
    PolyCoef_5   =   [0.00389150 , -0.083751 , -0.31082 , -1.5861];
    PolyCoef_6   =   [0.00303020 , -0.082676 , -0.48030];

    PolyCoef_7   =   [0.459 , -2.273];

    weights(n)   =   polyval(PolyCoef_1 , u);
    weights(1)   =   -weights(n);
    
    if n > 5
        weights(n-1) =   polyval(PolyCoef_2 , u);
        weights(2)   =   -weights(n-1);
    
        count  =   3;
        phi    =   (mtilde'*mtilde - 2 * mtilde(n)^2 - 2 * mtilde(n-1)^2) / ...
                (1 - 2 * weights(n)^2 - 2 * weights(n-1)^2);
    else
        count  =   2;
        phi    =   (mtilde'*mtilde - 2 * mtilde(n)^2) / ...
                (1 - 2 * weights(n)^2);
    end
        
    % Special attention when n = 3 (this is a special case).
    if n == 3
        % Royston (1992, p. 117)
        weights(1)  =   1/sqrt(2);
        weights(n)  =   -weights(1);
        phi = 1;
    end

    %
    % The vector 'WEIGHTS' obtained next corresponds to the same coefficients
    % listed by Shapiro-Wilk in their original test for small samples.
    %

    weights(count : n-count+1)  =  mtilde(count : n-count+1) / sqrt(phi);

    %
    % The Shapiro-Wilk statistic W is calculated to avoid excessive rounding
    % errors for W close to 1 (a potential problem in very large samples).
    %

    W   =   (weights' * x) ^2 / ((x - mean(x))' * (x - mean(x)));

    %
    % Calculate the normalized W and its significance level (exact for
    % n = 3). Royston (1992, p. 118) and Royston (1993b, p. 40, Table 1).
    %

    newn    =   log(n);

    if (n >= 4) && (n <= 11)
    
        mu      =   polyval(PolyCoef_3 , n);
        sigma   =   exp(polyval(PolyCoef_4 , n));    
        gam     =   polyval(PolyCoef_7 , n);
    
        newSWstatistic  =   -log(gam-log(1-W));
    
    elseif n > 11
    
        mu      =   polyval(PolyCoef_5 , newn);
        sigma   =   exp(polyval(PolyCoef_6 , newn));
    
        newSWstatistic  =   log(1 - W);
    
    elseif n == 3
        mu      =   0;
        sigma   =   1;
        newSWstatistic  =   0;
    end

    %
    % Compute the normalized Shapiro-Wilk statistic and its p-value.
    %

    NormalSWstatistic   =   (newSWstatistic - mu) / sigma;
    
    % NormalSWstatistic is referred to the upper tail of N(0,1),
    % Royston (1992, p. 119).
    pValue       =   1 - normcdf(NormalSWstatistic, 0, 1);
    
    % Special attention when n = 3 (this is a special case).
    if n == 3
        pValue  =   6/pi * (asin(sqrt(W)) - asin(sqrt(3/4)));
        % Royston (1982a, p. 121)
    end
    
end

%
% To maintain consistency with existing Statistics Toolbox hypothesis
% tests, returning 'H = 0' implies that we 'Do not reject the null 
% hypothesis at the significance level of alpha' and 'H = 1' implies 
% that we 'Reject the null hypothesis at significance level of alpha.'
%

H  = (alpha >= pValue);
end

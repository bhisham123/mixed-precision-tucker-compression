function T = hosvd1(X,tol,varargin)  
% HOSVD method 
% HOSVD method 
 
% Inputs
%   X: original tensor (d modes)
%   tol: desired tolerance

% Outputs
%   T: Tucker tensor


% Written by Bhisham Dev Verma, 2025


params = inputParser;
params.addParameter('ranks',[]);
params.parse(varargin{:});

ranks = params.Results.ranks;


% store dimensions and properties
sz = size(X);
d = length(sz);

normxsqr = collapse(X.^2);
eigsumthresh = tol.^2 * normxsqr / d;

% Pre allocate memory for factor matrices
U = cell(d,1);

t_mat = 0;
t_mult = 0;
t_fact = 0;
t_core = 0;

if ~isempty(ranks)
    if ~isvector(ranks) || length(ranks) ~= d
        error('Specified ranks must be a vector of length ndims(X)');
    end
    r = ranks;
else
    r = zeros(d,1);
end

for n = 1:d
    tic;
     M = double(tenmat(X,n));
    t_mat = t_mat + toc;

    tic;
    [~,R] = qr(M',0);
    [Q,S,~] = svd(R','econ');   % compute svd decomposition
    if r(n) == 0
        eigsum = cumsum(diag(S).^2,'reverse');
        r(n) = find(eigsum > eigsumthresh, 1, 'last');
    end
    % Extract factor matrix by picking out leading eigenvectors 
    U{n} = Q(:,1:r(n));
    t_fact = t_fact+toc;
end
%Compute core 
tic;
 G = ttm(X,U,'t');
t_core = t_core + toc;

% return Tucker tensor
T = ttensor(G,U);
times = [t_core,t_mult,t_fact,0, t_mat];
end



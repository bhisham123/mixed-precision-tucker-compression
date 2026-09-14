clc
clear all

%path to chop library
addpath("chop-master/")
%path to tensor toolbox
addpath("tensor_toolbox/")



%read datasets
dataset = "heat"
load('heat.mat');
X = tensor(Tt.data);
clear Tt;



%
if strcmp(dataset,"miranda")|| strcmp(dataset,"mhd")
    mix_prec_vec = ["3prec", "sb"];
else
    mix_prec_vec = ["7prec", "dsb", "ds"];
end



for  iprec = 1:length(mix_prec_vec)
    mix_prec = mix_prec_vec(iprec);
    switch mix_prec
        case "ds"
            u     = [2^-53, 2^-24];
            prec  = ['d','s'];
            bytes = [8,4];    
        case "dsh"
            u     = [2^-53, 2^-24, 2^-11]; 
            prec  = ['d','s','h'];
            bytes = [8,4,2];  
        case "dsb"
            u     = [2^-53, 2^-24, 2^-8]; 
            prec  = ['d','s','b'];
            bytes = [8,4,2];  
        case "dh"
            u     = [2^-53, 2^-11];
            prec  = ['d','h'];
            bytes = [8,2];
        case "db"
            u     = [2^-53, 2^-8];
            prec  = ['d','b'];
            bytes = [8,2];
        case "sh"
            u     = [2^-24, 2^-11];
            prec  = ['s','h'];
            bytes = [4,2];
        case "sb"
            u     = [2^-24, 2^-8];
            prec  = ['s','b'];
            bytes = [4,2]; 
        case "dt"
            u     = [2^-53, 2^-11];
            prec  = ['d','t'];
            bytes = [8,2];  % not really 2 bytes !
        case "7prec"
            u     = [2^-53, 2^-45, 2^-37, 2^-29, 2^-24, 2^-16, 2^-8];
            prec  = ["d","rp56","rp48","rp40","s","rp24","b"];
            bytes = [8,7,6,5,4,3,2];
        case "3prec"
            u     = [2^-24, 2^-16, 2^-8];
            prec  = ["s","rp24","b"];
            bytes = [4,3,2];
        otherwise
            error("Invalid choice of mix_prec");
    end

    normX = norm(X);
    
    p = length(u);
    d = length(size(X));
    
    if strcmp(dataset,"miranda")||strcmp(dataset,"mhd")
        eps_vec = [1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2];
    else
        eps_vec = 10.^(ceil(log10(u(1)))+1:-2);      
    end
    
    ratio.X.full = zeros(length(eps_vec),1);
    ratio.X.MP = zeros(length(eps_vec),1);
    ratio.X.coreFactorMP = zeros(length(eps_vec),1);
    
    ratio.Tcore.coreMP = zeros(length(eps_vec),1);
    
    ratio.T.coreMP = zeros(length(eps_vec),1);
    ratio.T.coreFactorMP = zeros(length(eps_vec),1);
    ratio.T.Tt = zeros(length(eps_vec),1);
    
    
    ratio.Ttcore.coreMP = zeros(length(eps_vec),1);
    ratio.Tt.coreMP = zeros(length(eps_vec),1);
    ratio.Tt.coreFactorMP = zeros(length(eps_vec),1);
    
    err.full = zeros(length(eps_vec),1);
    err.T = zeros(length(eps_vec),1);
    err.coreMP.relT = zeros(length(eps_vec),1);
    err.coreMP.relX = zeros(length(eps_vec),1);
    err.coreFactorMP.relT = zeros(length(eps_vec),1);
    err.coreFactorMP.relX = zeros(length(eps_vec),1);

    count.coreMP = cell(length(eps_vec),1);
    count.coreFactorMP = cell(length(eps_vec),1);
    
    
    
    for ii = 1:length(eps_vec)
        epsilon  = eps_vec(ii);
        disp("Precision: " + mix_prec);
        disp("Tolerance: " + num2str(epsilon));
        disp(' ')

        %compute HOSVD with tol epsilon to get compression; no mixprecision
        Tt = hosvd1(X,epsilon);
        
        err.full(ii) = norm(X-full(Tt))/normX;

        mem.Tt.core = prod(size(Tt.core))*bytes(1);
        mem.Tt.fact = sum(cellfun(@numel, Tt.U))*bytes(1);
        mem.X = prod(size(X))*bytes(1);

        ratio.X.full(ii) = mem.X/(mem.Tt.core + mem.Tt.fact);
        size(Tt.core)
        disp('Full precision HOSVD')
        disp(['Error: ' num2str(err.full(ii))]);
        disp(' ')
        clear Tt;
    
        %First apply HOSVD with  tolerance  alpha*epsilon to get compressed tucker tensor  
        alpha = 1
        T = hosvd1(X,alpha*epsilon);
        
        normT = norm(T);
        
        err.T(ii) = norm(X -full(T))/normX;

        mem.T.core = prod(size(T.core))*bytes(1);
        mem.T.fact = sum(cellfun(@numel, T.U))*bytes(1);
        size(T.core)
        disp(['Error full precision HOSVD with tolerance alpha*eps: ' num2str(err.T(ii))]);
        disp(' ');

        %mix precision tolerace
        mixPrecTol = max(epsilon - err.T(ii),(1-alpha)*epsilon);
        disp(['Mix precision tolerance: ' num2str(mixPrecTol)])

        % compute partitioning indices
        idx = ones(d, p+1);

        % global tolerance
        globalTol = (mixPrecTol^2) * normX^2;
        modeBaseTol = globalTol / d;
        
        Tm = T;
        leftover_mode = 0;   % carry across modes
        for j = 1:d
            %mode level budget
            modeTol = modeBaseTol + leftover_mode;
            leftover_mode = 0;
            
         
            M = double(tenmat(T.core,j));

            [~,R] = qr(M',0);
            [Q{j},S,~] = svd(R','econ');
            % [Q,s,~] = svd(tenmat(T.core,j).data);
            Tm.U{j} = Tm.U{j}*Q{j};
            % Tm.core = ttm(Tm.core,Q,j,'t');
            clear M;

            s = diag(S);
            squaredSV = s.^2;

            idx(j,p+1) = length(s)+1;
            tempEnd = idx(j,p+1);
            
            % split mode budget into (p-1) parts
            precBaseTol = modeTol/(p-1);
            leftover_prec = 0;

            for i = p:-1:2
                
                % effective budget at this stage
                tmpTol = precBaseTol + leftover_prec;
                threshold =  tmpTol/u(i)^2;

                searchVec = cumsum(squaredSV(1:tempEnd-1),'reverse');

                ind = find(searchVec > threshold, 1, 'last');
                if isempty(ind)
                    % ------------------------------------
                    % nothing exceeds threshold
                    % full leftover goes forward
                    % ------------------------------------
                    idx(j, i) = 1;
                    leftover_prec = tmpTol-searchVec(1)*(u(i)^2);
                    tempEnd = 1;
                    break
                else
                    idx(j,i) = ind+1;
                    tempEnd = idx(j,i);

                    if ind < length(searchVec)
                        usedEnergy = searchVec(ind+1);
                    else
                        usedEnergy = 0;
                    end
                    % leftover after satisfying constraint
                    leftover_prec = tmpTol - usedEnergy*(u(i)^2);  
                end
            end 
            % carry leftover from mode to next mode
            leftover_mode = leftover_prec;   
        end
        clear s;
        idx

      
        Tm.core = ttm(Tm.core,Q,'t');
        clear Q;
        

        ranges = repmat({1:p}, 1, d);
        [G{1:d}] = ndgrid(ranges{:});
        
        totalComb = numel(G{1});
        
        % Build matrix of n-tuples
        I = zeros(totalComb, d);
        
        for r = 1:d
            I(:,r) = G{r}(:);
        end
        clear G;

        disp('Only core in Mixed Precision')
        clear T;

        core_ent_count = zeros(1,p);
    
        for i = 1:totalComb
            b = I(i,:);
            id = max(b);
            
            % -------- precision selection --------
            switch prec(id)
                case "t"
                    options.format = "c"; options.params = [11,127];  options.round = 1;
                case "rp56"
                    options.format = "c"; options.params = [45,1023]; options.round = 1;
                case "rp48"
                    options.format = "c"; options.params = [37,1023]; options.round = 1;
                case "rp40"
                    options.format = "c"; options.params = [29,1023]; options.round = 1;
                case "rp24"
                    options.format = "c"; options.params = [16,127];  options.round = 1;
                otherwise
                    options.format = prec(id);
                    if isfield(options, 'params')
                        options = rmfield(options, 'params');
                    end
                    options.round = 1;
            end
            
            % -------- build slices --------
            subs = cell(1,d);
            for r = 1:d
                subs{r} = idx(r,b(r)) : idx(r,b(r)+1)-1;
            end
            
            % -------- apply chop --------  
            if ~any(cellfun(@isempty, subs)) && (prod(b)>1)          
                if prod(cellfun(@numel, subs)) == 1
                    Tm.core(subs{:}) = chop(Tm.core(subs{:}), options);
                    core_ent_count(id) = core_ent_count(id) + 1;
                else
                    Tm.core(subs{:}) = tensor(chop(Tm.core(subs{:}).data, options));
                    core_ent_count(id) = core_ent_count(id) + prod(size(Tm.core(subs{:})));
                end
            end
            
        end
        clear I;
    
        
    
        %relative error w.r.t. X 
        disp('Computing relative error')
        err.coreMP.relX(ii) = norm(X - full(Tm))/normX;
    
        disp(['Relative error: ' num2str(err.coreMP.relX(ii))]);
    
        
        N = prod(size(Tm.core)); % total number of elements in cores
        
        core_ent_count(1) = N-sum(core_ent_count(2:end)); % core_ent_count contains core element in each precision
        count.coreMP{ii} = core_ent_count;

        mem.Tm.core = sum(core_ent_count.*bytes); %space taken by mixed precision core
        
        
        ratio.Tcore.coreMP(ii) = mem.T.core/mem.Tm.core;
        ratio.T.coreMP(ii) = (mem.T.core + mem.T.fact)/(mem.Tm.core + mem.T.fact); %core is only in mixed precision
        ratio.X.MP(ii) = (mem.X)/(mem.Tm.core + mem.T.fact);
    
        ratio.Ttcore.coreMP(ii) = mem.Tt.core/mem.Tm.core;
        ratio.Tt.coreMP(ii) = (mem.Tt.core + mem.Tt.fact)/(mem.Tm.core + mem.T.fact); %core is only in mixed precision (compression improvement factor)
    
        ratio.T.Tt(ii) = (mem.T.core + mem.T.fact)/(mem.Tt.core + mem.Tt.fact);

        disp(['Compression imporvement factor: ', num2str(ratio.Tt.coreMP(ii))])
        % disp(['nBytes(CoreFP)/nBytes(CoreMP)  = ', num2str(ratio.Tcore.coreMP(ii))]);
        % disp(['nBytes(Tucker(eps/2))/nBytes(CoreMP+FactFP) = ', num2str(ratio.T.coreMP(ii))]);
        disp(' ')


        disp('Both core and factor matrices are in Mixed Precision')
        Tm1 = Tm;
        clear Tm;


        fact_ent_count = zeros(1,p);
        for k = 2:p
             % -------- precision selection --------
            switch prec(k)
                case "t"
                    options.format = "c"; options.params = [11,127];  options.round = 1;
                case "rp56"
                    options.format = "c"; options.params = [45,1023]; options.round = 2;
                case "rp48"
                    options.format = "c"; options.params = [37,1023]; options.round = 2;
                case "rp40"
                    options.format = "c"; options.params = [29,1023]; options.round = 2;
                case "rp24"
                    options.format = "c"; options.params = [16,127];  options.round = 1;
                otherwise
                    options.format = prec(k);
                    if isfield(options, 'params')
                        options = rmfield(options, 'params');
                    end
                    options.round = 1;
            end
            
    
            for j = 1:d
                Tm1.U{j}(:,idx(j,k):idx(j,k+1)-1) = chop(Tm1.U{j}(:,idx(j,k):idx(j,k+1)-1), options);
                fact_ent_count(k) = fact_ent_count(k) + prod(size(Tm1.U{j}(:,idx(j,k):idx(j,k+1)-1)));
            end
        end
    
        err.coreFactorMP.relX(ii) = norm(X - full(Tm1))/normX;
    
        disp(['Relative error: ' num2str(err.coreFactorMP.relX(ii))]);
 
        mem.Tm1.core = mem.Tm.core; %space taken by mixed precision core
        mem.Tm1.fact = sum(fact_ent_count.*bytes); %space taken by mixed precision factor matrices 

        count.coreFactorMP{ii} =  count.coreMP{ii} + fact_ent_count;
    
        ratio.T.coreFactorMP(ii) = (mem.T.core + mem.T.fact)/(mem.Tm1.core + mem.Tm1.fact); 
        ratio.X.coreFactorMP(ii) = mem.X/(mem.Tm1.core + mem.Tm1.fact);
        ratio.Tt.coreFactorMP(ii) = ((mem.Tt.core + mem.Tt.fact))/(mem.Tm1.core + mem.Tm1.fact); %both core and factor matrices in mixed precision (compression improvement factor)

        disp(['Compression improvement factor: ' num2str(ratio.Tt.coreFactorMP(ii))]);
        disp(' ')
        
    end
    name = dataset +"_"+num2str(alpha)+"_"+ mix_prec + "_hosvd.mat";
    save(name, 'err', 'ratio', 'eps_vec', 'mix_prec', 'count', 'prec');
end % loop on mix_prec_vec



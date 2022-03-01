function BruteESB(Adjacency,options)
    arguments
        options.system string = 'Rossler'
        options.params (3,1) double = (0.2, 0.2, 9.0)
        options.Ka double = 0.17
        options.Kb double = 4.614
        options.pathname string = './Backbones/'
        options.filename string = 'BruteBackbone.txt' 
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    H = zeros(3); if options.H == 'x', H(1,1)=1; 
    elseif options.H =='y', H(2,2)=1;
    elseif options.H =='z', H(3,3)=1;
    else H = matrix(options.H); end
    
    [f,df] = get_system(options.system);
    params = options.params;
    interval = [options.Ka, options.Kb];
    A = load(Adjacency);
    Name = split(Adjacency,'.')(1);
    savefile = join([options.pathname, options.filename],'/')
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    
    
    
end


function [K, new_ratio]= Find_ratio(inputs)
N, ES, interval = inputs;
EdgeTable = table(ES,'VariableNames',{'EndNodes'});
K = graph(EdgeTable);
bins = conncomp(K);
if sum(bins)==N
    E = eig(Laplacian(K));
    C = Find_C(interval, E);
    if C
        new_ratio = E(end)/E(2);
    else
        new_ratio=NaN;
    end
else
    new_ratio = NaN;
end
end

function P = powersets(itemlist, nn)
% By Paulo Abelha
%
% Returns the powerset of set S
%
% S is a cell array
% P is a cell array of cell arrays
    n = numel(itemlist);
    x = 1:n;
    P = cell(1,2^n);
    p_ix = 2;
    a = combnk(x,nn);
    for j=1:size(a,1)
        P{p_ix} = S(a(j,:));
        p_ix = p_ix + 1;
    end
end

function [f,df] = get_system(system)
    if system == 'Rossler'
        f = @Rossler;
        df = @DF_Rossler;
    elseif system == 'Lorenz'
        f = @Lorenz;
        df = @DF_Lorenz;
    end
end        
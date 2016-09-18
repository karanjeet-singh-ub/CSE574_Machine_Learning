clear;
filename1='C:\Users\karanjeet\Downloads\train-images-idx3-ubyte\train-images.idx3-ubyte';
fp1 = fopen(filename1, 'rb');
assert(fp1 ~= -1, ['Could not open ', filename1, '']);
magic = fread(fp1, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename1, '']);
numImages = fread(fp1, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp1, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp1, 1, 'int32', 0, 'ieee-be');
train_images = fread(fp1, inf, 'unsigned char');
train_images = reshape(train_images, numCols, numRows, numImages);
train_images = permute(train_images,[1 2 3]);
fclose(fp1);
train_images = reshape(train_images, size(train_images, 1) * size(train_images, 2), size(train_images, 3));
train_images = double(train_images) / 255;

filename2='C:\Users\karanjeet\Downloads\train-labels-idx1-ubyte\train-labels.idx1-ubyte';
fp2 = fopen(filename2, 'rb');
assert(fp2 ~= -1, ['Could not open ', filename2, '']);
magic = fread(fp2, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename2, '']);
numLabels = fread(fp2, 1, 'int32', 0, 'ieee-be');
train_labels = fread(fp2, inf, 'unsigned char');
assert(size(train_labels,1) == numLabels, 'Mismatch in label count');
fclose(fp2);

filename3='C:\Users\karanjeet\Downloads\t10k-images-idx3-ubyte\t10k-images.idx3-ubyte';
fp3 = fopen(filename3, 'rb');
assert(fp3 ~= -1, ['Could not open ', filename3, '']);
magic = fread(fp3, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename3, '']);
numImages = fread(fp3, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp3, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp3, 1, 'int32', 0, 'ieee-be');
test_images = fread(fp3, inf, 'unsigned char');
test_images = reshape(test_images, numCols, numRows, numImages);
test_images = permute(test_images,[1 2 3]);
fclose(fp3);
test_images = reshape(test_images, size(test_images, 1) * size(test_images, 2), size(test_images, 3));
test_images = double(test_images) / 255;

filename4='C:\Users\karanjeet\Downloads\t10k-labels-idx1-ubyte\t10k-labels.idx1-ubyte';
fp4 = fopen(filename4, 'rb');
assert(fp4 ~= -1, ['Could not open ', filename4, '']);
magic = fread(fp4, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename4, '']);
numLabels = fread(fp4, 1, 'int32', 0, 'ieee-be');
test_labels = fread(fp4, inf, 'unsigned char');
assert(size(test_labels,1) == numLabels, 'Mismatch in label count');
fclose(fp4);

train_input=transpose(train_images);
test_input=transpose(test_images);

train_size=60000;
test_size=10000;
features=784;
k=10;

train_output_k_format = zeros(train_size,k);
for i = 1:train_size
    train_output_k_format(i,train_labels(i)+1)=1;
end

test_output_k_format = zeros(test_size,k);
for i = 1:test_size
    test_output_k_format(i,test_labels(i)+1)=1;
end

train_design_mat = train_input;
test_design_mat = test_input;

b_size = 30;
iterations = 20;
w0 = randn(features,k)*0.1;
n_a = 1;
n_b = 1;
w = w0;
rand_train = randperm(train_size);
counter = 0;
neta = n_a / (n_b+counter);
b_start = 1;
testlabels=zeros(1,test_size);
for i=1:test_size
    [~,testlabels(i)] = max(test_output_k_format(i,:));
end

while(counter < iterations)
   
    b_stop = min(train_size,b_start+b_size-1);
    curr_train_design_mat = train_design_mat(rand_train(b_start:b_stop),:);
    curr_train_output_k_format = train_output_k_format(rand_train(b_start:b_stop),:);
    curr_train_size = size(curr_train_design_mat,1);
    curr_train_output_k_format_size = size(curr_train_output_k_format,2);
    logit_err = zeros(size(w));
    
    for n = 1:curr_train_size
        pn = exp(w'*curr_train_design_mat(n,:)');
        yn = pn/sum(pn);
        for i=1:curr_train_output_k_format_size
            logit_err(:,i) = logit_err(:,i) + (yn(i) - curr_train_output_k_format(n,i))*curr_train_design_mat(n,:)';
        end
    end
    logit_err = (1/curr_train_size)*logit_err;
    dw = logit_err;
    w = w - neta*dw;
    b_start = b_start+b_size;
    
    if(b_start>train_size)
        trained_val=zeros(1,test_size);
        for i=1:test_size
            as = w'*test_design_mat(i,:)';
            [~,trained_val(i)] = max(as);
        end
        errorrate = 1 - sum(trained_val(:)==testlabels(:)) / size(test_design_mat,1);
        fprintf('Test error = %.1f%%, neta = %f, iteration = %d\n', errorrate*100, neta, counter+1);
        
        b_start = 1;
        counter = counter+1;
        neta = n_a / (n_b+counter);
        rand_train = randperm(train_size); 
    end
end
Wlr=w;
blr=zeros(1,10);

% neural single hidden layer

hidden_layer = 500;
neural_neta = 0.5;
neural_b_size = 100;
neural_b_start = 1;
neural_iterations = 5;
hidden_wts = rand(hidden_layer, features);
neural_out_wts = rand(k, hidden_layer);
hidden_wts = hidden_wts./size(hidden_wts, 2);
neural_out_wts = neural_out_wts./size(neural_out_wts, 2);
n = zeros(neural_b_size);
targetValues=transpose(train_output_k_format);

cc=0;
while(cc<neural_iterations)
    neural_b_stop = min(train_size,neural_b_start+neural_b_size-1);
    curr_train_design_mat = train_images(:,rand_train(neural_b_start:neural_b_stop));
    curr_train_output_k_format = targetValues(:,rand_train(neural_b_start:neural_b_stop));
    curr_train_size = size(curr_train_design_mat,1);
    curr_train_output_k_format_size = size(curr_train_output_k_format,2);
        
    for j = 1: neural_b_size

        train_line = curr_train_design_mat(:, j);
        hidden_inp = hidden_wts*train_line;
        hidden_out = 1./(1 + exp(-hidden_inp));
        second_inp = neural_out_wts*hidden_out;
        neural_out_line = 1./(1 + exp(-second_inp));

        train_out_line = curr_train_output_k_format(:, j);

        second_err = (1./(1 + exp(-second_inp)).*(1 - 1./(1 + exp(-second_inp)))).*(neural_out_line - train_out_line);
        hidden_err = (1./(1 + exp(-hidden_inp)).*(1 - 1./(1 + exp(-hidden_inp)))).*(neural_out_wts'*second_err);

        neural_out_wts = neural_out_wts - neural_neta.*second_err*hidden_out';
        hidden_wts = hidden_wts - neural_neta.*hidden_err*train_line';
    end;
    neural_b_start = neural_b_start+neural_b_size;

    if(neural_b_start>train_size)

        trained_val=zeros(1,test_size);
        test_input = test_images;
        for i=1:test_size
            test_inp_line = test_input(:, i);
            neural_out_line = 1./(1 + exp(-(neural_out_wts*(1./(1 + exp(-(hidden_wts*test_inp_line)))))));
            [~,trained_val(i)] = max(neural_out_line);
        end
        errorrate = 1 - sum(trained_val(:)==testlabels(:)) / size(test_design_mat,1);
        fprintf('Test error = %.1f%%, iteration = %d\n', errorrate*100, cc+1);
        
        neural_b_start = 1;
        cc = cc+1;
        rand_train = randperm(train_size);
    end
end;

Wnn1=transpose(hidden_wts);
Wnn2=transpose(neural_out_wts);
bnn1=zeros(1,hidden_layer);
bnn2=zeros(1,k);
h='sigmoid';
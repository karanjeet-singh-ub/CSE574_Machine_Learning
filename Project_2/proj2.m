% had parsed the files using python and not matlab

% ------------------------------------------------------------
% ------------------------------------------------------------

% training_input1   ( Real world data training input features, 46 X 55698), starting from 1
% training1_output  ( Real World Data training output vector, 1 X 55698), starting from 1
% validation_input  ( Real World Data validation input vector, 46 X 13926),starting from 55699
% validation_output ( Real World Data validation output vector, 1 X 13926),starting from 55699
% syn_train_inp     ( Synthetic Data Input features 10 X 1600),starting from 1
% syn_train_out     ( Systhetic Data Output features 1 X 1600),starting from 1
% syn_valid_inp     ( Systhetic Data validation input features 10 X 400),starting from 1601
% syn_valid_out     ( Systhetic Data validation output features 1 X 400),starting from 1601

% ------------------------------------------------------------
% ------------------------------------------------------------



% closed form solution (for real data):

trainInd1=zeros(55698,1);
for j=1:55698
    trainInd1(j)=j;
end

trainInd2=zeros(1600,1);
for j=1:1600
    trainInd2(j)=j;
end

validInd1=zeros(6923,1);
for j=55699:62660
validInd1(j-55698) = j;
end

validInd2=zeros(400,1);
for j=1601:2000
validInd2(j-1600) = j;
end

sigma1=diag(var(training_input1))+eye(46)*.3;
M1=6;
Sigma1=zeros(46,46,M1);
for j=1:M1
    Sigma1(:,:,j)=sigma1;
end
for M1=6            % had used a for loop earier to get the best combnation
m1=randi([1 55698],1,M1);
mu1=zeros(M1,46);

for j=1:M1
    mu1(j,:)=training_input1(m1(j),:);
end
design_matrix=zeros(55698,M1);

for i = 1:55698
    for j= 1:M1
        if(j==1)
            design_matrix(i,j) = 1;
        else
            design_matrix(i,j)=exp(((-0.5)*((training_input1(i,:)-mu1(j,:))*(inv(sigma1)*transpose((training_input1(i,:)-mu1(j,:)))))));
        end
    end
end

lambda1=0.3;
for l=0.3;
lmda=eye(M1)*l;
w1=(inv((transpose(design_matrix)*design_matrix) + lmda))*(transpose(design_matrix)*training1_output);
sqared_error=(transpose(training1_output) - (transpose(w1)*transpose(design_matrix))).^2;
trainPer1 = sqrt(sum(sqared_error)/55698);
end
end

validation_design_matrix=zeros(6923,M1);
for i = 1:6923
    for j= 1:M1
        if(j==1)
            validation_design_matrix(i,j) = 1;
        else
            validation_design_matrix(i,j)=exp(((-0.5)*((validation_input(i,:)-mu1(j,:))*(inv(sigma1)*transpose((validation_input(i,:)-mu1(j,:)))))));
        end
    end
end

validation_sqared_error=(transpose(validation_output) - (transpose(w1)*transpose(validation_design_matrix))).^2;
validPer1 = sqrt(sum(validation_sqared_error)/6923);


% closed form solution for Synthetic data:
M2=6;

sigma2=diag(var(syn_train_inp))+eye(10)/10000;
for j=1:M2
    Sigma1(:,:,j)=sigma1;
end

for M2=6
m1=randi([1 1600],1,M2);
mu2=zeros(M2,10);

for j=1:M2
    mu2(j,:)=syn_train_inp(m1(j),:);
end
syn_design_mat=zeros(1600,M2);

for i = 1:1600
    for j= 1:M2
        if(j==1)
            syn_design_mat(i,j) = 1;
        else
            syn_design_mat(i,j)=exp(((-0.5)*((syn_train_inp(i,:)-mu2(j,:))*(inv(syn_sigma)*transpose((syn_train_inp(i,:)-mu2(j,:)))))));
        end
    end
end

lambda2=6;
for l=6;            % was using a for loop to the best combination
lmda=eye(M2)*l;
w2=(inv((transpose(syn_design_mat)*syn_design_mat) + lmda))*(transpose(syn_design_mat)*syn_train_out);
syn_sqared_error=(transpose(syn_train_out) - (transpose(w2)*transpose(syn_design_mat))).^2;
trainPer2 = sqrt(sum(syn_sqared_error)/1600);
end
end

syn_valid_design_matrix=zeros(400,M2);
for i = 1:400
    for j= 1:M2
        if(j==1)
            syn_valid_design_matrix(i,j) = 1;
        else
            syn_valid_design_matrix(i,j)=exp(((-0.5)*((syn_valid_inp(i,:)-mu2(j,:))*(inv(syn_sigma)*transpose((syn_valid_inp(i,:)-mu2(j,:)))))));
        end
    end
end

syn_validation_sqared_error=(transpose(syn_valid_out) - (transpose(w2)*transpose(syn_valid_design_matrix))).^2;
validPer2 = sqrt(sum(syn_validation_sqared_error)/400);




% stochastic solution for real world data

for up=1.25
for down=0.2
for nn = 1.2
    w01=[5;5;5;5;5;5];
    n1=nn;
    iterations_1=55698;
    dw1=zeros(6,iterations_1);
    eta1=zeros(1,iterations_1);
    for j=1:55698
        delta_w_1=((n1*(((training1_output(j) - (transpose(w01)*transpose(design_matrix(j,:))))*transpose(design_matrix(j,:)))-(0.3*w01))));
        updated_wt_1 = w01+delta_w_1;
        eta1(:,j)=n1;
        dw1(:,j)=delta_w_1;
        if sqrt(sum((updated_wt_1).^2)-sum((w01).^2))<0.0001
            n1=n1*up;
        else
            n1=n1*down;
        end
        w01=updated_wt_1;
    end
    stochastic_sqared_error_1=(transpose(training1_output) - (transpose(w01)*transpose(design_matrix))).^2;
    stocastic_error_1 = sqrt(sum(stochastic_sqared_error_1)/55698);
    w01=[5;5;5;5;5;5];
end
end
end


% stochastic solution for synthetic data

for up=1.75
for down=.59
for nn = 1.8
    w02=[5;5;5;5;5;5];
    n2=nn;
    iterations_2=1600;
    dw2=zeros(6,iterations_2);
    eta2=zeros(1,iterations_2);
    for j=1:1600
        delta_w_2=((n2*(((syn_train_out(j) - (transpose(w02)*transpose(syn_design_mat(j,:))))*transpose(syn_design_mat(j,:)))-(6*w02))));
        updated_wt_2 = w02+delta_w_2;
        eta2(:,j)=n2;
        dw2(:,j)=delta_w_2;
        if sqrt(sum((updated_wt_2).^2)-sum((w02).^2))<0.0001
            n2=n2*up;
        else
            n2=n2*down;
        end
        w02=updated_wt_2;
    end
    stochastic_sqared_error_2=(transpose(syn_train_out) - (transpose(w02)*transpose(syn_design_mat))).^2;
    stocastic_error_2 = sqrt(sum(stochastic_sqared_error_2)/1600);
    w02=[5;5;5;5;5;5];
end
end
end

mu1=transpose(mu1);
mu2=transpose(mu2);
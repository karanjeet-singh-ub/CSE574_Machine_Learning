xls_file = xlsread('university data.xlsx');
UBitName=['k';'a';'r';'a';'n';'j';'e';'e'];
personNumber=['5';'0';'1';'6';'9';'9';'1';'6'];
x=[xls_file(:,3),xls_file(:,4),xls_file(:,5),xls_file(:,6)];
total_mean=mean(x);
mu1=total_mean(1);
mu2=total_mean(2);
mu3=total_mean(3);
mu4=total_mean(4);
total_var=var(x);
var1=total_var(1);
var2=total_var(2);
var3=total_var(3);
var4=total_var(4);
total_sigma = std(x);
sigma1=total_sigma(1);
sigma2=total_sigma(2);
sigma3=total_sigma(3);
sigma4=total_sigma(4);
covarianceMat=cov(x);
correlationMat=corrcoef(x);
logLikelihood=0;
log1= sum(log(normpdf(x(:,1),total_mean(1),total_sigma(1))));
log2= sum(log(normpdf(x(:,2),total_mean(2),total_sigma(2))));
log3= sum(log(normpdf(x(:,3),total_mean(3),total_sigma(3))));
log4= sum(log(normpdf(x(:,4),total_mean(4),total_sigma(4))));
logLikelihood = log1 + log2 + log3 + log4;
a=[0 0 0 0;0 0 1 0;0 1 0 0;0 1 1 0;1 0 0 0;1 0 1 0;1 1 0 0;1 1 1 0];
b=[0 0 0 0;0 0 0 1;0 1 0 1;1 0 0 0;1 0 0 1;1 1 0 0;1 1 0 1;0 1 0 0];
c=[0 0 0 0;0 0 0 1;0 0 1 0;0 0 1 1;1 0 0 0;1 0 0 1;1 0 1 0;1 0 1 1];
d=[0 0 0 0;0 0 0 1;0 0 1 0;0 0 1 1;0 1 0 0;0 1 0 1;0 1 1 0;0 1 1 1];
m=1;
n=0;
r=[];
bigmat=cell(1,1000);
total_prob=0;
for i = 1:8
    for j = 1:8
        for k = 1:8
            for l = 1:8
                s=[d(i,:);c(j,:);b(k,:);a(l,:)];
                if ((s(1,2) + s(2,1) == 2) | (s(1,3) + s(3,1) == 2) | (s(1,4) + s(4,1) == 2) | (s(2,3) + s(3,2) == 2) | (s(2,4) + s(4,2) == 2) | (s(3,4) + s(4,3) == 2) | (s(1,2) + s(2,3) + s(3,1)== 3) | (s(2,1) + s(3,2) + s(1,3)== 3) | (s(1,2) + s(2,4) + s(4,1)== 3) | (s(2,1) + s(4,2) + s(1,4)== 3) | (s(1,3) + s(3,4) + s(4,1)== 3) | (s(3,1) + s(4,3) + s(1,4)== 3) | (s(2,3) + s(3,4) + s(4,2)== 3) | (s(3,2) + s(4,3) + s(2,4)== 3) | (s(1,2) + s(2,4) + s(4,3) + s(3,1) == 4) | (s(1,3) + s(3,4) + s(4,2) + s(2,1) == 4))
                    n=0;
                else
                    bigmat{m} = s;
                    m=m+1;
                    col_1_depd=find(s(:,1));
                    col_2_depd=find(s(:,2));
                    col_3_depd=find(s(:,3));
                    col_4_depd=find(s(:,4));
                    if isempty(col_1_depd)
                        prob1 = log1;
                    else
                        col_1=[1;col_1_depd];
                        prob1=(sum(log(mvnpdf(x(:,col_1),total_mean(col_1),covarianceMat(col_1,col_1))))) - (sum(log(mvnpdf(x(:,col_1_depd),total_mean(col_1_depd),covarianceMat(col_1_depd,col_1_depd)))));
                    end
                    if isempty(col_2_depd)
                        prob2 = log2;
                    else
                        col_2=[2;col_2_depd];
                        prob2=(sum(log(mvnpdf(x(:,col_2),total_mean(col_2),covarianceMat(col_2,col_2))))) - (sum(log(mvnpdf(x(:,col_2_depd),total_mean(col_2_depd),covarianceMat(col_2_depd,col_2_depd)))));
                    end
                    if isempty(col_3_depd)
                        prob3 = log3;
                    else
                        col_3=[3;col_3_depd];
                        prob3=(sum(log(mvnpdf(x(:,col_3),total_mean(col_3),covarianceMat(col_3,col_3))))) - (sum(log(mvnpdf(x(:,col_3_depd),total_mean(col_3_depd),covarianceMat(col_3_depd,col_3_depd)))));
                    end
                    if isempty(col_4_depd)
                        prob4 = log4;
                    else
                        col_4=[4;col_4_depd];
                        prob4=(sum(log(mvnpdf(x(:,col_4),total_mean(col_4),covarianceMat(col_4,col_4))))) - (sum(log(mvnpdf(x(:,col_4_depd),total_mean(col_4_depd),covarianceMat(col_4_depd,col_4_depd)))));
                    end
                    total_prob=prob1+prob2+prob3+prob4;
                    r(end+1) = total_prob;
                end
            end
        end
    end
end
BNlogLikelihood=max(r);
index=find(r==max(r),1);
BNgraph=bigmat{index};
/********************************************************************************/
/*  Elie Weintraub                                                              */
/*  AI -  Project #2 - Neural Network Project                                   */
/*                                                                              */ 
/*  test.cpp  - Neural Network Testing Program                                  */
/*                                                                              */
/********************************************************************************/

#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <cstring>

#define A 0
#define B 1
#define C 2
#define D 3

#define SIG(x) (1/(1+exp(-(x))))
#define F1(precision,recall) (2*(precision)*(recall)/((precision)+(recall)))

using namespace std;

//Loads trained neural net weights from text file
void loadWeights(int &Ni,int &Nh, int &No, double **&Wji, double **&Wkj){
    string netFileName;
	//Open file for reading
	cout<<"Enter the name of the trained neural network file: ";
	cin>>netFileName;
    ifstream netFile(netFileName.c_str());
    if (!netFile){ 
        cerr << "Error: Unable to open "<<netFileName<<endl;
        exit(1);
    }
	//Read in Ni,Nh, and No
	netFile>>Ni>>Nh>>No;
	//Allocate weight matrices: Wji (input i to hidden node j) and Wkj (hidden node j to output k)
	Wji=new double*[Nh];
	for(int j=0;j<Nh;j++){ 
		Wji[j] = new double[Ni+1];
	}
	Wkj=new double*[No];
	for(int k=0;k<No;k++){ 
		Wkj[k] = new double[Nh+1];
	}
	//Read in weights
	for(int j=0;j<Nh;j++){ 
		for(int i=0;i<Ni+1;i++){
			netFile>>Wji[j][i];
		}
	}
	for(int k=0;k<No;k++){ 
		for(int j=0;j<Nh+1;j++){
			netFile>>Wkj[k][j];
		}
	}
	//Close file
	netFile.close();	
}

//Load in test set from text file
void loadTestingExamples(int &Nn, double **&Xni, int **&Ynk){
	string testingFileName;
	int Ni,No;
	//Open file for reading
	cout<<"Enter the name of the testing set file: ";
	cin>>testingFileName;
    ifstream testingFile(testingFileName.c_str());
    if (!testingFile){ 
        cerr << "Error: Unable to open "<<testingFileName<<endl;
        exit(1);
    }
	//Read in Nn, Ni, and No
	testingFile>>Nn>>Ni>>No; //Nn: number of testing examples
	//Allocate input and output matrices: Xni (input i for example n) and Ynk (output k for example n)
	Xni=new double*[Nn];
	for(int n=0;n<Nn;n++){ 
		Xni[n] = new double[Ni];
	}
	Ynk=new int*[Nn];
	for(int n=0;n<Nn;n++){ 
		Ynk[n] = new int[No];
	}
	//Read in inputs and outputs
	for(int n=0;n<Nn;n++){ 
		for(int i=0;i<Ni;i++){
			testingFile>>Xni[n][i];
		}
		for(int k=0;k<No;k++){
			testingFile>>Ynk[n][k];
		}
	}
	//Close file
	testingFile.close();
}

//Calculate in_x for node x
double calcIn(double * const weights, double * const a_in,int n_weights){
	double in_x=0;
	for(int i=0;i<n_weights;i++){
		in_x+=weights[i]*a_in[i];
	}
	return in_x;
}

//Test the neural net using the testing set and compute Confusion matrix	
void testNeuralNet(int ** &confusionMat,double ** const Xni,int ** const Ynk,
                   double **Wji,double **Wkj,int Ni,int Nh,int No,int Nn){
	
	double *in_j=new double[Nh];
	double *in_k=new double[No];
	double *a_i=new double[Ni+1];
	double *a_j=new double[Nh+1];
	double *a_k=new double[No];

	//Initialize Confusion Matrix
	confusionMat=new int*[No];
	for(int k=0;k<No;k++){ 
		confusionMat[k] = new int[4];
	}
	for(int k=0;k<No;k++){ 
		for(int i=0;i<4;i++){
			confusionMat[k][i]=0;
		}
	}
	
	//Test
	for(int n=0;n<Nn;n++){
		//Calculate Input Layer Activations
		a_i[0]=-1;
		for(int i=0;i<Ni;i++){
			a_i[i+1]=Xni[n][i];
		}	
		//Calculate Hidden Layer Activations
		a_j[0]=-1;
		for(int j=0;j<Nh;j++){	
			in_j[j]=calcIn(Wji[j],a_i,Ni+1);
			a_j[j+1]=SIG(in_j[j]);
		}
		//Calculate Output Layer Activations and Confusion Matrix
		for(int k=0;k<No;k++){	
			in_k[k]=calcIn(Wkj[k],a_j,Nh+1);
			a_k[k]=round(SIG(in_k[k]));
			switch(Ynk[n][k]){
				case 0:
					(Ynk[n][k]==a_k[k])?confusionMat[k][D]++:confusionMat[k][B]++;
					break;
				case 1:
					(Ynk[n][k]==a_k[k])?confusionMat[k][A]++:confusionMat[k][C]++;
					break;
				default:
					cerr<<"We should never get here!"<<endl;
			}		
		}
	}    
	//Clean up
	free(in_j);free(in_k);
	free(a_i);free(a_j);free(a_k);
}

//Compute micro-averaged accuracy
double microAccuracy(int ** const confusionMat, int No){
	int a=0,b=0,c=0,d=0;
	for(int k=0;k<No;k++){
		a+=confusionMat[k][A]; b+=confusionMat[k][B];
		c+=confusionMat[k][C]; d+=confusionMat[k][D];
	}
	return (double)(a+d)/(a+b+c+d);
}

//Compute micro-averaged precision 
double microPrecision(int ** const confusionMat, int No){
	int a=0,b=0,c=0,d=0;
	for(int k=0;k<No;k++){
		a+=confusionMat[k][A]; b+=confusionMat[k][B];
	}
	return (double)a/(a+b);
}

//Compute micro-averaged recall
double microRecall(int ** const confusionMat, int No){
	int a=0,b=0,c=0,d=0;
	for(int k=0;k<No;k++){
		a+=confusionMat[k][A]; c+=confusionMat[k][C]; 
	}
	return (double)a/(a+c);
}

//Compute macro-averaged accuracy and accuracy vector
double macroAccuracy(int ** const confusionMat,double * &acc_vec,int No){
	int a,b,c,d;
	double acc=0;
	acc_vec=new double[No];
	
	for(int k=0;k<No;k++){
		a=confusionMat[k][A]; b=confusionMat[k][B];
		c=confusionMat[k][C]; d=confusionMat[k][D];
		acc_vec[k]=(double)(a+d)/(a+b+c+d);
		acc+=acc_vec[k];
	}
	return  acc/No;
}

//Compute macro-averaged precision and precision vector
double macroPrecision(int ** const confusionMat,double * &prec_vec,int No){
	int a,b,c,d;
	double prec=0;
	prec_vec=new double[No];
	for(int k=0;k<No;k++){
		a=confusionMat[k][A]; b=confusionMat[k][B];
		prec_vec[k]=(double)a/(a+b);
		prec+=prec_vec[k];
	}
	return  prec/No;
}

//Compute macro-averaged recall and recall vector
double macroRecall(int ** const confusionMat,double * &rec_vec,int No){
	int a,b,c,d;
	double rec=0;
	rec_vec=new double[No];
	for(int k=0;k<No;k++){
		a=confusionMat[k][A]; c=confusionMat[k][C]; 
		rec_vec[k]=(double)a/(a+c);
		rec+=rec_vec[k];
	}
	return  rec/No;
}

//Compute F1 vector
void F1Vector(double * const prec_vec,double * const rec_vec,double * & F1_vec,int No){
	F1_vec=new double[No];
	for(int k=0;k<No;k++){
		F1_vec[k]=F1(prec_vec[k],rec_vec[k]);
	}
}
	
//Get results  and write to a text file
void getResults(int No, int ** const confusionMat,double * &acc_vec,
                double * &prec_vec,double * &rec_vec,double * &F1_vec){
				
	//Get Results
	double micro_acc=microAccuracy(confusionMat,No);
	double micro_prec=microPrecision(confusionMat,No);
	double micro_rec=microRecall(confusionMat,No);
	double macro_acc=macroAccuracy(confusionMat,acc_vec,No);
	double macro_prec=macroPrecision(confusionMat,prec_vec,No);
	double macro_rec=macroRecall(confusionMat,rec_vec,No);
	F1Vector(prec_vec,rec_vec,F1_vec,No);
	double micro_F1=F1(micro_prec,micro_rec);
	double macro_F1=F1(macro_prec,macro_rec);

	//Open file for writing
	string resultsFileName;
	cout<<"Enter a name for the results file: ";
	cin>>resultsFileName;
    ofstream resultsFile(resultsFileName.c_str());
    if (!resultsFile){ 
        cerr << "Error: Unable to open "<<resultsFileName<<endl;
        exit(1);
    }
	
	//Write out results
	resultsFile<<fixed<<setprecision(3);
	//Category specific results
	for(int k=0;k<No;k++){ 
		for(int i=0;i<4;i++){
			resultsFile<<confusionMat[k][i]<<" ";
		}
		resultsFile<<acc_vec[k]<<" "<<prec_vec[k]<<" "<<rec_vec[k]<<" "<<F1_vec[k]<<endl;
	}
	//Micro-averaged results
	resultsFile<<micro_acc<<" "<<micro_prec<<" "<<micro_rec<<" "<<micro_F1<<endl;
	//Macro-averaged results
	resultsFile<<macro_acc<<" "<<macro_prec<<" "<<macro_rec<<" "<<macro_F1<<endl;
	
	//Close file
	resultsFile.close();
}

//Clean up 
void CleanUp(double **&Wji,double **&Wkj,double **&Xni,int **&Ynk,int **&confusionMat,
             double *&acc_vec,double *&prec_vec, double *&rec_vec,double *&F1_vec,
			 int Nh,int No,int Nn){
	for(int j=0;j<Nh;j++)
		delete[] Wji[j];
	delete[] Wji;
	
	for(int k=0;k<No;k++) 
		delete[] Wkj[k];
	delete[] Wkj;
	
	for(int n=0;n<Nn;n++) 
		delete[] Xni[n];
	delete[] Xni;
	
	for(int n=0;n<Nn;n++) 
		delete[] Ynk[n];
	delete[] Ynk;
	
	for(int k=0;k<No;k++) 
		delete[] confusionMat[k];
	delete[] confusionMat;
	
	delete acc_vec; delete prec_vec;
	delete rec_vec; delete F1_vec;
}	

int main(){
	int Ni,Nh,No,Nn,**Ynk,**confusionMat;
	double **Wji,**Wkj,**Xni,*acc_vec,*prec_vec,*rec_vec,*F1_vec;
	
	loadWeights(Ni,Nh,No,Wji,Wkj);
	loadTestingExamples(Nn,Xni,Ynk);
	testNeuralNet(confusionMat,Xni,Ynk,Wji,Wkj,Ni,Nh,No,Nn);
	getResults(No,confusionMat,acc_vec,prec_vec,rec_vec,F1_vec);
	CleanUp(Wji,Wkj,Xni,Ynk,confusionMat,acc_vec,prec_vec,rec_vec,F1_vec,Nh,No,Nn);
	
	return 0;
}
	
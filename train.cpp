/********************************************************************************/
/*  Elie Weintraub                                                              */
/*  AI -  Project #2 - Neural Network Project                                   */
/*                                                                              */ 
/*  train.cpp  - Train a neural network using back propagation                  */
/*                                                                              */
/********************************************************************************/

#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <cstring>

#define SIG(x) (1/(1+exp(-(x))))
#define SIG_PRIME(x) (SIG((x))*(1-SIG((x))))
using namespace std;

//Loads initial neural net weights from text file
void loadInitialWeights(int &Ni,int &Nh, int &No, double **&Wji, double **&Wkj){
    string netFileName;
	//Open file for reading
	cout<<"Enter the name of the initial neural network file: ";
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

//Load in training examples from text file
void loadTrainingExamples(int &Nn, double **&Xni, int **&Ynk){
	string trainingFileName;
	int Ni,No;
	//Open file for reading
	cout<<"Enter the name of the training set file: ";
	cin>>trainingFileName;
    ifstream trainingFile(trainingFileName.c_str());
    if (!trainingFile){ 
        cerr << "Error: Unable to open "<<trainingFileName<<endl;
        exit(1);
    }
	//Read in Nn, Ni, and No
	trainingFile>>Nn>>Ni>>No; //Nn: number of training examples
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
			trainingFile>>Xni[n][i];
		}
		for(int k=0;k<No;k++){
			trainingFile>>Ynk[n][k];
		}
	}
	//Close file
	trainingFile.close();
}

//Calculate in_x for node x
double calcIn(double * const weights, double * const a_in,int n_weights){
	double in_x=0;
	for(int i=0;i<n_weights;i++){
		in_x+=weights[i]*a_in[i];
	}
	return in_x;
}

//Calculate back-propagation term at node j
double calcBackProp(double ** const Wkj,double * const delta_k,int No, int node_j){
	double backProp_j=0;
	for(int k=0;k<No;k++){
		backProp_j+=Wkj[k][node_j]*delta_k[k];
	}
	return backProp_j;
}

//Train the neural net using the training set	
void trainNeuralNet(double ** const Xni,int ** const Ynk, double **Wji,
                    double **Wkj,int Ni,int Nh,int No,int Nn){
	int n_epochs;
	double alpha; //learning rate
	
	double *in_j=new double[Nh];
	double *in_k=new double[No];
	double *a_i=new double[Ni+1];
	double *a_j=new double[Nh+1];
	double *a_k=new double[No];
	double *delta_j=new double[Nh];
	double *delta_k=new double[No];
	 
	//Prompt user for number of epochs and learning rate
	cout<<"Enter the number of epochs: ";
	cin>>n_epochs;
	cout<<"Enter the learning rate: ";
	cin>>alpha;

	for(int epoch=0;epoch<n_epochs;epoch++){
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
			//Calculate Output Layer Activations
			for(int k=0;k<No;k++){	
				in_k[k]=calcIn(Wkj[k],a_j,Nh+1);
				a_k[k]=SIG(in_k[k]);
			}
			//Calculate Output Layer Errors
			for(int k=0;k<No;k++){
				delta_k[k]=SIG_PRIME(in_k[k])*(Ynk[n][k]-a_k[k]);
			}
			//Calculate Hidden layer Errors
			for(int j=0;j<Nh;j++){
				delta_j[j]=SIG_PRIME(in_j[j])*calcBackProp(Wkj,delta_k,No,j+1);
			}
			//Recalculate Wji
			for(int j=0;j<Nh;j++){ 
				for(int i=0;i<Ni+1;i++){
					Wji[j][i]+=alpha*a_i[i]*delta_j[j];
				}
			}
			//Recalculate Wkj
			for(int k=0;k<No;k++){ 
				for(int j=0;j<Nh+1;j++){
					Wkj[k][j]+=alpha*a_j[j]*delta_k[k];
				}
			}
		}          //end example loop
	}           //end epoch loop	
	//Clean up
	free(in_j);free(in_k);
	free(a_i);free(a_j);free(a_k);
	free(delta_j);free(delta_k);
}

//Write trained neural net to a text file
void writeFinalWeights(int Ni,int Nh, int No, double ** const Wji, double ** const Wkj){
	string trainedFileName;
	//Open file for writing
	cout<<"Enter a name for the trained neural network file: ";
	cin>>trainedFileName;
    ofstream trainedFile(trainedFileName.c_str());
    if (!trainedFile){ 
        cerr << "Error: Unable to open "<<trainedFileName<<endl;
        exit(1);
    }
	//Write out trained neural net
	int i,j;
	trainedFile<<Ni<<" "<<Nh<<" "<<No<<endl;
	trainedFile<<fixed<<setprecision(3);
	for(int j=0;j<Nh;j++){ 
		for(i=0;i<Ni;i++){
			trainedFile<<Wji[j][i]<<" ";
		}
		trainedFile<<Wji[j][i]<<endl;
	}
	for(int k=0;k<No;k++){ 
		for(j=0;j<Nh;j++){
			trainedFile<<Wkj[k][j]<<" ";
		}
		trainedFile<<Wkj[k][j]<<endl;
	}
	//Close file
	trainedFile.close();
}

//Clean up 
void CleanUp(double **&Wji,double **&Wkj,double **&Xni,int **&Ynk,int Nh,int No,int Nn){
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
}	

int main(){
	int Ni,Nh,No,Nn,**Ynk;
	double **Wji,**Wkj,**Xni;
	
	loadInitialWeights(Ni,Nh,No,Wji,Wkj);
	loadTrainingExamples(Nn,Xni,Ynk);
	trainNeuralNet(Xni,Ynk,Wji,Wkj,Ni,Nh,No,Nn);
	writeFinalWeights(Ni,Nh,No,Wji,Wkj);
	CleanUp(Wji,Wkj,Xni,Ynk,Nh,No,Nn);
	
	return 0;
}
	
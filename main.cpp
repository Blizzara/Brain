#include <iostream>

//#define DEBUG

//#include "outputlayer.h"
//#include "inputlayer.h"

#include "neuralnetwork.h"
#include "reader.h"

int main(int argc, char **argv) {
/*    NeuralNetwork nn(Eigen::Vector4d(3,3,3,1), 0.2, 0.1);

    typedef Eigen::Matrix<double,1,1> Vector1d;
    Vector1d rtrue;
    rtrue << 1;
    Vector1d rfalse;
    rfalse << 0;

    int NUM_ITERS = 100000;


    std::cout << "Beginning learning phase...\niteration 0";
    
    
    
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data;
    training_data.push_back(std::make_pair<Eigen::VectorXd, Eigen::VectorXd>(Eigen::Vector3d(1,1,1), rtrue));
    training_data.push_back(std::make_pair<Eigen::VectorXd, Eigen::VectorXd>(Eigen::Vector3d(1,0,0), rtrue));
    training_data.push_back(std::make_pair<Eigen::VectorXd, Eigen::VectorXd>(Eigen::Vector3d(0,1,0), rtrue));
    training_data.push_back(std::make_pair<Eigen::VectorXd, Eigen::VectorXd>(Eigen::Vector3d(0,0,1), rtrue));
    training_data.push_back(std::make_pair<Eigen::VectorXd, Eigen::VectorXd>(Eigen::Vector3d(1,1,0), rfalse));
    training_data.push_back(std::make_pair<Eigen::VectorXd, Eigen::VectorXd>(Eigen::Vector3d(0,1,1), rfalse));
    training_data.push_back(std::make_pair<Eigen::VectorXd, Eigen::VectorXd>(Eigen::Vector3d(1,0,1), rfalse));
    training_data.push_back(std::make_pair<Eigen::VectorXd, Eigen::VectorXd>(Eigen::Vector3d(0,0,0), rfalse));
    
    nn.TrainBatch(training_data, 100000, 0.000001);
    
    std::cout <<"Learning done."<< std::endl;
    std::cout <<"Beginning testing phase...\n";

    std::cout << nn.Evaluate(Eigen::Vector3d(1,1,1)) << " vs " << 1 << std::endl;
    std::cout << nn.Evaluate(Eigen::Vector3d(1,1,0)) << " vs " << 0 << std::endl;
    std::cout << nn.Evaluate(Eigen::Vector3d(1,0,0)) << " vs " << 1 << std::endl;
    std::cout << nn.Evaluate(Eigen::Vector3d(1,0,1)) << " vs " << 0 << std::endl;
    std::cout << nn.Evaluate(Eigen::Vector3d(0,1,1)) << " vs " << 0 << std::endl;
    std::cout << nn.Evaluate(Eigen::Vector3d(0,1,0)) << " vs " << 1 << std::endl;
    std::cout << nn.Evaluate(Eigen::Vector3d(0,0,1)) << " vs " << 1 << std::endl;
    std::cout << nn.Evaluate(Eigen::Vector3d(0,0,0)) << " vs " << 0 << std::endl;
*/

    Reader reader;
    Eigen::VectorXd layers(4);
    layers << 10, 10, 5,1;
    NeuralNetwork nn(layers, 0.2, 0.1);
    
    std::cout << "Reading training data..." << std::flush;
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data;
    std::cout << reader.ReadTrainingData("training_sample.csv",10,1,training_data) << " lines read." << std::endl;
    
    std::cout << "Beginning learning phase..." << std::endl;
    nn.TrainBatch(training_data, 50000, 0.001);
    std::cout << "End training phase..." << std::endl;
    nn.PrintWeights();
    std::cout << "Reading evaluation data..." << std::flush;
    std::vector<Eigen::VectorXd> evaluation_input_data;
    std::vector<Eigen::VectorXd> evaluation_output_data;
    
    int evaluation_rowcount = 0;
    std::cout << (evaluation_rowcount = reader.ReadEvaluationData("training_normalized.csv",10,1,evaluation_input_data, evaluation_output_data)) << " lines read." << std::endl;
    std::cout << "Beginning testing phase..." << std::endl;
    double cumulative_mse = 0;
    int success_count = 0;
    double success_square_error_threshold = 0.16;
    for(int i = 0; i < evaluation_rowcount; ++i) {
      Eigen::VectorXd output = nn.Evaluate(evaluation_input_data[i]);
      Eigen::VectorXd error = evaluation_output_data[i] - output;
      double square_error = error.cwiseProduct(error).sum() / error.size();
      cumulative_mse += square_error;
      if(square_error < success_square_error_threshold) {
	++success_count;
      }
      
      //std::cout << nn.Evaluate(evaluation_input_data[i]) << " vs. " << evaluation_output_data[i] << std::endl;
    }
    
    nn.PrintWeights();
    nn.PrintStates();
    
    
    std::cout << "End testing phase. Mean square error " << cumulative_mse/evaluation_rowcount << "\n";
    std::cout << success_count << " / " << evaluation_rowcount << " rows classified correctly (mse < " << success_square_error_threshold << ")\n";

    //std::cout << nn.Evaluate(evaluation_input_data[0]) << " vs. " << evaluation_output_data[0] << std::endl;
    //nn.PrintStates();
    //std::cout << nn.Evaluate(evaluation_input_data[1]) << " vs. " << evaluation_output_data[1] << std::endl;
    //nn.PrintStates();
    //std::cout << nn.Evaluate(evaluation_input_data[4]) << " vs. " << evaluation_output_data[4] << std::endl;
    //nn.PrintStates();
    
    return 0;
}

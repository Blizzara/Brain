/*
    Copyright (c) 2013, <copyright holder> <email>
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the <organization> nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY <copyright holder> <email> ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL <copyright holder> <email> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include "neuralnetwork.h"

#include <assert.h>
#include <algorithm>
#include <iomanip>


//#if __GNUC__MINOR__ < 7
//typedef std::chrono::monotonic_clock steady_clock;
//#else
typedef std::chrono::steady_clock steady_clock;
//#endif

NeuralNetwork::NeuralNetwork(Eigen::VectorXd node_sizes, float learning_coeff, float momentum_coeff)
{
    learning_coeff_ = learning_coeff;
    momentum_coeff_ = momentum_coeff;

    mse_ = 0;
    mse_count_ = 0;
    
    layer_count = 0;
    
    for(int i = 0; i < node_sizes.size(); ++i) {
        layers_.push_back(Layer(node_sizes(i), i));
        ++layer_count;
    }

    for(int i = 1; i < layer_count; ++i) {
        weights_.push_back(Eigen::MatrixXd::Random(node_sizes(i-1),node_sizes(i)));
        previous_change_.push_back(weights_[i-1]);
    }


    /*    for(int i = 0; i < layer_count-1; ++i) {
            std::cout <<"\n" << i << "\n" << weights_[i] << std::endl;
        }
    */
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& other)
{

}

NeuralNetwork::~NeuralNetwork()
{

}

NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& other)
{
    return *this;
}

bool NeuralNetwork::operator==(const NeuralNetwork& other) const
{
///TODO: return ...;
}



int NeuralNetwork::Propagate(int from_index)
{
#ifdef DEBUG
    std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << " begins" << std::endl;
    //std::cout << "Nodes^T:\n" << nodes_.transpose() << std::endl;
    //std::cout << "Weights:\n" << weights_ << std::endl;
#endif

    Layer& from = layers_[from_index];
    Layer& to = layers_[from_index+1];
    Eigen::MatrixXd& weights = weights_[from_index];

    const Eigen::VectorXd states = from.GetNodeStates();
    const Eigen::VectorXd next_states = states.transpose()*weights;
    const Eigen::VectorXd nonlinear_states = Activation(next_states);

    to.SetNodeStates(nonlinear_states);

#ifdef DEBUG
    std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << " ends" << std::endl;
#endif
}


int NeuralNetwork::BackPropagate(int from_index)
{
#ifdef DEBUG
    std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << " begins" << std::endl;
    //std::cout << "Previous^T:\n" << previous.transpose() << std::endl;
    //std::cout << "Weights^T:\n" << weights_.transpose() << std::endl;
#endif

    assert(from_index > 0 && from_index < layer_count);
    Layer& from = layers_[from_index];
    Layer& to = layers_[from_index-1];
    Eigen::MatrixXd& weights = weights_[from_index-1];

    const VectorXd& prev_error = from.GetError();
    VectorXd error = prev_error.transpose()*weights.transpose();
    to.SetError(error);
    //VectorXd delta = DActivation(error);

#ifdef DEBUG

    std::cout << "Weights: \n" << weights << "\n";
    std::cout << "Error:" << prev_error.transpose() << "\n";

    std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << " ends " << std::endl;
#endif

    //return error;
}

int NeuralNetwork::RecalculateWeights(int index)
{
#ifdef DEBUG
    std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << " begins" << std::endl;
    //std::cout << "deltas^T:\n" << deltas.transpose() << std::endl;
#endif

    Layer& left = layers_[index];
    Layer& right = layers_[index+1];

    VectorXd error = right.GetError();
    VectorXd df = DActivation(right.GetNodeStates());
    VectorXd coeff = error.cwiseProduct(df)*learning_coeff_;

    MatrixXd change = left.GetNodeStates()*coeff.transpose();
    MatrixXd& previous_change = previous_change_[index];
    MatrixXd change_by_momentum = momentum_coeff_*previous_change;

    previous_change_[index] = change;

    weights_[index] += change;
    weights_[index] += change_by_momentum;

#ifdef DEBUG
    std::cout << "h: " << h << " error^T: " << error.transpose() << " df^T: " << df.transpose() << std::endl;
    std::cout << "coeff^T: " << coeff.transpose() << std::endl;
    std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << " ends - change is " << change.transpose() << std::endl;
#endif

}


int NeuralNetwork::Train(VectorXd input, VectorXd output)
{
#ifdef DEBUG
    std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << " begins" << std::endl;
    //std::cout << "deltas^T:\n" << deltas.transpose() << std::endl;
#endif

    VectorXd simulated_output = Evaluate(input);
    VectorXd error = output-simulated_output;


    VectorXd error_squared = error.cwiseProduct(error);
    double square_error = 1.0 * error_squared.sum()/error_squared.size() ;
    mse_ += square_error;
    ++mse_count_;

    Layer& last = layers_[layer_count-1];
    last.SetError(error);

    for(int i = layer_count-1; i > 0; --i) {
        BackPropagate(i);
    }

#ifdef DEBUG
    std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << " recalculate " << std::endl;
#endif

    for(int i = 0; i < layer_count-1; ++i) {
        RecalculateWeights(i);
    }


#ifdef DEBUG
    std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << " ends " << std::endl;
#endif

}

VectorXd NeuralNetwork::Evaluate(VectorXd input)
{
#ifdef DEBUG
    std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << " begins" << std::endl;
    //std::cout << "deltas^T:\n" << deltas.transpose() << std::endl;
#endif
    Layer& first = layers_[0];
    first.SetNodeStates(input);

#ifdef DEBUG
    std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << " propagate " << std::endl;
#endif

    for(int i = 0; i < layer_count-1; ++i) {
        Propagate(i);
    }

    Layer& last = layers_[layer_count-1];
    VectorXd output = last.GetNodeStates();
    return output;


#ifdef DEBUG
    std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << " ends " << std::endl;
#endif

}

double ActivationD(double y) {
    return 1/(1+exp(-y));
}

VectorXd NeuralNetwork::Activation(VectorXd new_node_states)
{
    return new_node_states.unaryExpr(std::ptr_fun(ActivationD));
}

double DActivationD(double y) {
    return y*(1-y);
}


VectorXd NeuralNetwork::DActivation(VectorXd new_node_states)
{
    return new_node_states.unaryExpr(std::ptr_fun(DActivationD));;
}


int NeuralNetwork::TrainBatch(std::vector<std::pair<Eigen::VectorXd,Eigen::VectorXd>> input_output_list, long int max_iterations, double min_mse_threshold)
{
    int i = 0;

    if(max_iterations < 0 && min_mse_threshold < 0) {
        std::cout << "No max number of iterations or a threshold specified, using defaults (max=10000, thres=0.0001)" << std::endl;
        max_iterations = 10000;
        min_mse_threshold = 0.0001;
    }

    int progress_update_interval = max_iterations > 0 ? (max_iterations) : 0;
    if(max_iterations > 0 && progress_update_interval > max_iterations/100) {
        progress_update_interval = max_iterations/100;
    }
    if(progress_update_interval < 1) {
        progress_update_interval = 1;
    }

    double mse_update_threshold = 2.0;
    double previous_mse = 1;
    int previous_iterations = 0;
    int iterations_per_second = 0;
    

    steady_clock::time_point first_time_point_ = steady_clock::now();
    steady_clock::time_point prev_time_point_ = steady_clock::now();

    while(true) {
        ++i;
        steady_clock::time_point now = steady_clock::now();
        double ms_since_previous = std::chrono::duration_cast<std::chrono::milliseconds>(now-prev_time_point_).count();
        double mse = GetMSE();
        std::cout << std::setiosflags(std::ios::fixed);
        if(ms_since_previous > 1000) {

            int s_since_start = std::chrono::duration_cast<std::chrono::seconds>(now-first_time_point_).count();
            int secs = s_since_start % 60;
            int mins = (s_since_start / 60)%60;
            int hours = (s_since_start) / 3600;

            double percent_complete = i*100.0/max_iterations;
            if( iterations_per_second == 0) iterations_per_second = i-previous_iterations;
	    else iterations_per_second = iterations_per_second*2.0/3.0+(i-previous_iterations)/3.0;
            int estimate_seconds_left = (max_iterations - i)/iterations_per_second;
            int est_secs = estimate_seconds_left % 60;
            int est_mins = (estimate_seconds_left / 60)%60;
            int est_hours = estimate_seconds_left / 3600;

            double mse_improvement_ratio = mse/previous_mse;


            std::cout << std::setprecision(2) << "\riteration " << i << "/" << max_iterations << " (" << percent_complete << "%, " << iterations_per_second << " iter/s), ";
            std::cout << std::setprecision(4) << "mse=" << mse << "/" << std::resetiosflags(std::ios::fixed) <<  min_mse_threshold << std::setiosflags(std::ios::fixed) << " (" << mse_improvement_ratio << "), ";
            std::cout << std::setprecision(2) << "time: ";
            if(hours > 0) std::cout << hours << "h ";
            if(mins > 0) std::cout << mins << "min ";
            std::cout << secs << "s, ";
            std::cout << "ETA: ";
            if(est_hours > 0) std::cout << est_hours << "h ";
            if(est_mins > 0) std::cout << est_mins << "min ";
            std::cout << est_secs << "s ";
            std::cout << std::flush;
            previous_mse = mse;
            prev_time_point_ = now;
            previous_iterations  = i;
        }
        /*
            if(min_mse_threshold > 0 && max_iterations > 0 && i % progress_update_interval == 0) {
                std::cout << "\riteration " << i << "/" << max_iterations << " (" << i*100/max_iterations << "%), mse=" << mse << "/" << min_mse_threshold;
            std::cout << " (" << mse/previous_mse << "), time elapsed since beginning " << "                " << std::flush;
            }
            else if(max_iterations > 0 && i % progress_update_interval == 0) {
                std::cout << "\riteration " << i << "/" << max_iterations << " (" << i*100/max_iterations << "%), mse=" << mse;
            std::cout << " (" << mse/previous_mse << ")" << "                " << std::flush;
            }
            else if (min_mse_threshold > 0 && mse/previous_mse > mse_update_threshold) {
                std::cout << "\riteration " << i <<  ", mse=" << mse  << "/" << min_mse_threshold;
            std::cout << " (" << mse/previous_mse << ")" << "                " << std::flush;
            }
            */


        if(max_iterations > 0 && i > max_iterations) {
            std::cout << "\nIteration limit reached, stopping learning.\n";
            return 0;
        }
        else if(min_mse_threshold > 0 && mse < min_mse_threshold) {
            std::cout << "\nMSE threshold reached, stopping learning.\n";
            return 0;
        }

        ClearMSE();
        std::random_shuffle(input_output_list.begin(), input_output_list.end());
#pragma omp paraller for
	for(std::pair<Eigen::VectorXd,Eigen::VectorXd> pair : input_output_list) {
            Train(pair.first, pair.second);
        }
    }
}



void NeuralNetwork::PrintWeights()
{
    for(int i = 0; i < layer_count-1; ++i)
    {
        std::cout << "Matrix " << i << "\n" << weights_[i] << std::endl;
    }
}

void NeuralNetwork::PrintStates()
{
    for(int i = 0; i < layer_count; ++i)
    {
        std::cout << "Layer " << i << ": " << (layers_[i]).GetNodeStates().transpose() << std::endl;
    }
}


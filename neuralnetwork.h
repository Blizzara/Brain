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


#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <chrono>

#include <eigen3/Eigen/Dense>

#include "layer.h"


class NeuralNetwork
{

public:
    NeuralNetwork(Eigen::VectorXd node_sizes, float learning_coeff, float momentum_coeff);

    virtual ~NeuralNetwork();
    virtual bool operator==(const NeuralNetwork& other) const;

    int TrainBatch(std::vector<std::pair<Eigen::VectorXd,Eigen::VectorXd>> input_output_list, long max_iterations, double min_mse_threshold);

    int Train(VectorXd input, VectorXd output);
    VectorXd Evaluate(VectorXd input);

    void PrintWeights();
    void PrintStates();

    double GetMSE() const {
        if(mse_ <= 0 || mse_count_ == 0) {
            return 1;
        }
        else return mse_/mse_count_;
    };
    void ClearMSE() {

        mse_ = 0;
        mse_count_ = 0;
    };
private:

    std::vector<Layer> layers_;
    std::vector<Eigen::MatrixXd> weights_;
    std::vector<Eigen::MatrixXd> previous_change_;

    int layer_count;
    float learning_coeff_;
    float momentum_coeff_;

    double mse_;
    int mse_count_;

    NeuralNetwork(const NeuralNetwork& other);
    virtual NeuralNetwork& operator=(const NeuralNetwork& other);

    VectorXd Activation(VectorXd new_node_states);
//  double ActivationD(double y);
    VectorXd DActivation(VectorXd new_node_states);
//  double DActivationD(double y);

    int Propagate(int from_index);
    int BackPropagate(int from_index);
    int RecalculateWeights(int index);

};

#endif // NEURALNETWORK_H

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



#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <eigen3/Eigen/Dense>

typedef Eigen::VectorXd VectorXd;
typedef Eigen::MatrixXd MatrixXd;

class Layer
{

public:
    Layer(int size, int index);
    //Layer(const Layer& other);
    
    virtual ~Layer();
    virtual bool operator==(const Layer& other) const;
    
    int size() const { return size_; };
    int SetNodeStates(const VectorXd new_node_states);
    const VectorXd& GetNodeStates() const { return nodes_; };
    
    int SetError(const VectorXd new_error);
    const VectorXd& GetError() const { return error_; };
    virtual Layer& operator=(const Layer& other);
private:
    
    
    VectorXd nodes_;
    VectorXd error_;
    const int size_;
    int index_;
    
    
};

#endif // LAYER_H

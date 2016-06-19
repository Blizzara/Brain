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


#include "layer.h"

Layer::Layer(int size, int index) : size_(size)
{
#ifdef DEBUG
  std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << " begins - layer " << index_ << std::endl;
#endif
  //nodes_ = Eigen::VectorXd::Zero(size_);
  nodes_.setZero(size_);
  error_.setZero(size_);
  index_ = index;
#ifdef DEBUG
  std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << " ends - layer " << index_ << std::endl;
#endif
}

/*
Layer::Layer(const Layer& other)
{

}
*/

Layer::~Layer()
{

}

Layer& Layer::operator=(const Layer& other)
{
    return *this;
}

bool Layer::operator==(const Layer& other) const
{
    return (this)==(&other);
}


int Layer::SetNodeStates(const VectorXd new_node_states)
{
#ifdef DEBUG
  std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << ":" << index_ << " begins" << std::endl;
  //std::cout << "deltas^T:\n" << deltas.transpose() << std::endl;
#endif
  nodes_ = new_node_states;
#ifdef DEBUG
  std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << ":" << index_ << " ends" << std::endl;
  //std::cout << "deltas^T:\n" << deltas.transpose() << std::endl;
#endif
}

int Layer::SetError(const VectorXd new_error)
{
#ifdef DEBUG
  std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << ":" << index_ << " begins : error^T " << new_error.transpose() << std::endl;
  //std::cout << "deltas^T:\n" << deltas.transpose() << std::endl;
#endif
  error_ = new_error;
#ifdef DEBUG
  std::cout << __FILE__ << ":" << __LINE__ << " - " << __FUNCTION__ << ":" << index_ << " ends" << std::endl;
  //std::cout << "deltas^T:\n" << deltas.transpose() << std::endl;
#endif
}





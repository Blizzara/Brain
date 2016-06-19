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


#include "reader.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

Reader::Reader()
{

}

Reader::Reader(const Reader& other)
{

}

Reader::~Reader()
{

}

Reader& Reader::operator=(const Reader& other)
{
    return *this;
}

bool Reader::operator==(const Reader& other) const
{
///TODO: return ...;
}


int Reader::ReadTrainingData(const std::string& filename, int input_count, int output_count, std::vector< std::pair<Eigen::VectorXd, Eigen::VectorXd> >& training_data)
{
    std::ifstream  data(filename);
    //data.open(filename,std::ifstream::in);

    std::string line;
    int rows = 0;
    while(std::getline(data,line))
    {

//      std::cout << "read " << rows << " line: " << line << std::endl;
        Eigen::VectorXd input(input_count);
        Eigen::VectorXd output(output_count);
        std::stringstream  lineStream(line);
        std::string        cell;
        try {

            for(int i = 0; i < input_count && std::getline(lineStream,cell,','); ++i)
            {
                double d = std::stod(cell);
//          std::cout << "read " << i << " value: " << cell << " | " << d << std::endl;
                input[i] = d;
            }
            for(int i = 0; i < input_count && std::getline(lineStream,cell,','); ++i)
            {
                double d = std::stod(cell);
//          std::cout << "read " << i << " value: " << cell << " | " << d << std::endl;
                output[i] = d;
            }
        } catch (...) {
            continue;
        }
        ++rows;
        training_data.push_back(std::pair<Eigen::VectorXd, Eigen::VectorXd>(input,output));
    }
    return rows;
}

int Reader::ReadEvaluationData(const std::string& filename, int input_count, int output_count, std::vector<Eigen::VectorXd >& input_data, std::vector<Eigen::VectorXd >& output_data)
{
    std::ifstream  data(filename);
    //data.open(filename,std::ifstream::in);

    std::string line;
    int rows = 0;
    while(std::getline(data,line))
    {

//      std::cout << "read " << rows << " line: " << line << std::endl;
        Eigen::VectorXd input(input_count);
        Eigen::VectorXd output(output_count);
        std::stringstream  lineStream(line);
        std::string        cell;
        try {
            for(int i = 0; i < input_count && std::getline(lineStream,cell,','); ++i)
            {
                double d = std::stod(cell);
//          std::cout << "read " << i << " value: " << cell << " | " << d << std::endl;
                input[i] = d;
            }
            for(int i = 0; i < input_count && std::getline(lineStream,cell,','); ++i)
            {
                double d = std::stod(cell);
//          std::cout << "read " << i << " value: " << cell << " | " << d << std::endl;
                output[i] = d;
            }
        } catch (...) {
            continue;
        }
        ++rows;
        input_data.push_back(input);
        output_data.push_back(output);
    }
    return rows;
}

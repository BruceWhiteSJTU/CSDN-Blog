#pragma once
#include<vector>
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

template<typename int dimIn, typename int dimOut>
class LMOptimizerLeastSquares
{
public:
	typedef Matrix<double, dimIn, 1> InputType;
	typedef Matrix<double, dimOut, 1> OutputType;
	typedef Matrix<double, dimOut, dimIn> JacobianType;

	class objectTerm
	{
	public:
		virtual OutputType function(InputType X) = 0;
		virtual JacobianType Jacobian(InputType X) = 0;
	};

	void addTerm(objectTerm* term)
	{
		terms.push_back(term);
	}
	virtual void ConstructJacobian()
	{
		int N = terms.size();
		MatrixXd J(N*dimOut, dimIn);
		for (int k = 0; k < N; k++)
		{
			J.block(k*dimOut, 0, dimOut, dimIn) = terms[k]->Jacobian(X_);
		}
		Jacobian_ = J;
	}
	virtual void Constructb()
	{
		int N = terms.size();
		VectorXd b(N*dimOut);
		for (int k = 0; k < N; k++)
		{
			b.segment(k*dimOut, dimOut) = terms[k]->function(X_);
		}
		b_ = b;
	}
	virtual void ComputeUpdate()
	{
		Matrix<double, dimIn, dimIn> A = Jacobian_.transpose()*Jacobian_;
		Matrix<double, dimIn, 1> g = -Jacobian_.transpose()*b_;
		Matrix<double, dimIn, dimIn> M = A + lambda * MatrixXd(A.diagonal().asDiagonal());
		InputType delta = M.colPivHouseholderQr().solve(g);
		delta_ = delta;
	}
	virtual InputType Compose(InputType X, InputType delta)
	{
		return (X + delta);
	}
	virtual double TotalError(InputType X)
	{
		double sum = 0;
		for (int k = 0; k < terms.size(); k++)
		{
			sum += terms[k]->function(X).squaredNorm();
		}
		return 0.5*sum;
	}
	InputType Optimize(InputType X0, int niters)
	{
		lambda = 1e-4;
		X_ = X0;
		for (int k = 0; k < niters; k++)
		{
			ConstructJacobian();
			Constructb();
			ComputeUpdate();

			if (delta_.norm() < 1e-10)
			{
				break;
			}

			if (TotalError(Compose(X_, delta_)) < TotalError(X_))
			{
				X_ = Compose(X_, delta_);
				lambda /= 10;
			}
			else
			{
				lambda *= 10;
			}
			cout << "iteration = " << k << " , error = " << TotalError(X_) << " , lambda = " << lambda << endl;
		}
		return X_;
	}

	vector<objectTerm*> terms;
	InputType X_;
	InputType delta_;
	MatrixXd Jacobian_;
	VectorXd b_;
	double lambda;
};
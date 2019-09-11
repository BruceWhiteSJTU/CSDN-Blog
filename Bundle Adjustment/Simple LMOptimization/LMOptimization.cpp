#include<iostream>
#include<vector>
#include<random>
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Dense>
#include"LMOptimizerLeastSquares.h"

using namespace std;
using namespace Eigen;

class LMOptimizer2d
{
public:
	LMOptimizer2d()
	{
		x = vector<double>();
		y = vector<double>();
	}
	double function(Vector2d X)
	{
		double m = X[0], c = X[1];
		int N = x.size();
		double sum = 0;
		for (int k = 0; k < N; k++)
		{
			double f = exp(m*x [k]+ c) - y[k];
			sum += f * f;
		}
		return 0.5*sum;
	}
	MatrixXd Jacobian(Vector2d X)
	{
		int N = x.size();
		double m = X[0], c = X[1];
		MatrixXd J(N, 2);
		for (int k = 0; k < N; k++)
		{
			double f = exp(m*x[k] + c);
			J(k, 0) = f * x[k];
			J(k, 1) = f;
		}
		return J;
	}
	VectorXd constructb(Vector2d X)
	{
		int N = x.size();
		double m = X[0], c = X[1];
		VectorXd b(N);
		for (int k = 0; k < N; k++)
		{
			b[k] = exp(m*x[k] + c) - y[k];
		}
		return b;
	}

	Vector2d optimize(Vector2d x0,int niter)
	{
		double lambda = 1e-4;
		Vector2d X = x0;
		for (int k = 0; k < niter; k++)
		{
			MatrixXd A = Jacobian(X);
			VectorXd b = constructb(X);
			Matrix2d M = A.transpose()*A;
			M = M + lambda * Matrix2d(M.diagonal().asDiagonal());
			Vector2d delta = M.colPivHouseholderQr().solve(Vector2d(-A.transpose()*b));
			if (delta.norm() < 1e-10)
			{
				break;
			}
			if (function(X + delta) < function(X))
			{
				X = X + delta;
				lambda /= 10;
			}
			else
			{
				lambda *= 10;
			}
			cout << "iteration = "<<k<<" , error = " << function(X) << " , lambda = " << lambda << endl;
		}
		return X;
	}
	void addData(double x, double y)
	{
		this->x.push_back(x);
		this->y.push_back(y);
	}
	Vector2d X;
	vector<double> x;
	vector<double> y;
};

class ErrorTerm :public LMOptimizerLeastSquares<2, 1>::objectTerm
{
public:
	ErrorTerm(double x, double y)
	{
		this->x = x;
		this->y = y;
	}
	virtual Matrix<double,1,1> function(Vector2d X)
	{
		double m = X[0], c = X[1];
		double f = exp(m*x + c) - y;
		return Matrix<double,1,1>(f);
	}
	virtual Matrix<double,1,2> Jacobian(Vector2d X)
	{
		Vector2d J;
		double m = X[0], c = X[1];
		double f = exp(m*x + c);
		J[0] = f * x;
		J[1] = f;
		return J.transpose();
	}

	double x, y;
};
int main()
{
	default_random_engine e;
	normal_distribution<double> n(0,0.1);
	double m = 0.3, c = 0.1;
	vector<pair<double, double>> data;
	for (double x = 0.0; x < 5.0; x += 0.05)
	{
		double y = exp(m*x + c)+n(e);
		data.push_back(make_pair(x, y));
	}

	LMOptimizer2d opt;
	LMOptimizerLeastSquares<2, 1> optls;
	for (int k = 0; k < data.size(); k++)
	{
		opt.addData(data[k].first, data[k].second);

		ErrorTerm* e = new ErrorTerm(data[k].first, data[k].second);
		optls.addTerm(e);
	}

	Vector2d result=opt.optimize(Vector2d::Zero(), 50);
	cout << "result : m = " << result[0] << " , c = " << result[1] << endl;
	Vector2d resultls = optls.Optimize(Vector2d::Zero(), 50);
	cout << "result2 : m = " << resultls[0] << " , c = " << resultls[1] << endl;
	return 0;
}
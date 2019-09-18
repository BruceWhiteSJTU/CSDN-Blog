#pragma once
#include<vector>
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Dense>
#include<eigen3/Eigen/Geometry>
#include"se3quat.h"

using namespace std;
using namespace Eigen;
using SE3Type::SE3Quat;

class ReprojectionError
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	ReprojectionError() :fx(0), fy(0), cx(0), cy(0), u(0), v(0), camIdx(0), pointIdx(0) {}
	void setMeasurement(double u, double v)
	{
		this->u = u;
		this->v = v;
	}
	void setCameraParam(double fx, double fy, double cx, double cy)
	{
		this->fx = fx;
		this->fy = fy;
		this->cx = cx;
		this->cy = cy;
	}
	Vector2d function(const SE3Quat& pose, const Vector3d& p);
	Matrix<double, 2, 9> Jacobian(const SE3Quat & pose, const Vector3d & p);

	int camIdx, pointIdx;
private:
	Matrix3d skew(const Vector3d& phi);

	double fx, fy, cx, cy;
	double u, v;
};

class LMOptimizationBA
{
public:
	typedef Matrix<double, 6, 6> JacobianCC;
	typedef Matrix<double, 3, 3> JacobianPP;
	typedef Matrix<double, 6, 3> JacobianCP;
	typedef Matrix<double, 2, 9> JacobianType;
	typedef Matrix<double, 6, 1> Vector6d;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	LMOptimizationBA() {}
	void addCamera(SE3Quat cam)
	{
		cameras.push_back(cam);
	}
	void addPoint(Vector3d p)
	{
		points.push_back(p);
	}
	void addTerm(ReprojectionError* term)
	{
		terms.push_back(term);
	}
	void InitializeStorage(int numCams, int numPoints, int numObs)
	{
		cameras.reserve(numCams);
		points.reserve(numPoints);
		terms.reserve(numObs);
	}
	void Optimize(int niters);
	void SaveKeyFrameTrajectoryToFile(const char* file, vector<double> TimeStamps);

	// parameters
	vector<SE3Quat> cameras;
	vector<Vector3d> points;
	// error terms
	vector<ReprojectionError*> terms;
private:
	void PrepareOptimization();
	void ConstructJacobian();
	void ComputeUpdate();
	void ComputeNewParams();
	double TotalError(const vector<SE3Quat>& cameras, const vector<Vector3d>& points) const;

	// jacobians
	vector<JacobianCC> Hcc;
	vector<JacobianPP> Hpp;
	vector<vector<JacobianCP>> Hcp;
	VectorXd bc;
	VectorXd bp;
	// increments
	vector<Vector6d> deltacs;
	vector<Vector3d> deltaps;
	double lambda;
	// new params
	vector<SE3Quat> camerasUpdated;
	vector<Vector3d> pointsUpdated;
};
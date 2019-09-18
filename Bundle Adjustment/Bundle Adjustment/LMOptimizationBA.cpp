#include"LMOptimizationBA.h"
#include<iostream>
#include<fstream>
#include<iomanip>
#include<omp.h>

Vector2d ReprojectionError::function(const SE3Quat& pose, const Vector3d& p)
{
	Vector2d error;

	Vector3d pc = pose * p;
	double up = fx * pc[0] / pc[2];
	double vp = fy * pc[1] / pc[2];

	error[0] = up - u;
	error[1] = vp - v;

	return error;
}
Matrix<double, 2, 9> ReprojectionError::Jacobian(const SE3Quat & pose, const Vector3d & p)
{
	Matrix<double, 2, 9> J;

	Vector3d pc = pose * p;
	double x = pc[0], y = pc[1], z = pc[2];
	double z2 = z * z;

	Matrix<double, 2, 3> J_e_pc;
	J_e_pc << fx / z, 0, -fx * x / z2,
		0, fy / z, -fy * y / z2;

	Matrix<double, 3, 6> J_pc_ksi;
	//J_pc_ksi << Matrix3d::Identity(), -skew(pc);
	J_pc_ksi << -skew(pc), Matrix3d::Identity();

	Matrix<double, 2, 6> J_e_ksi = J_e_pc * J_pc_ksi;

	Matrix3d J_pc_p = pose.rotation().toRotationMatrix();
	Matrix<double, 2, 3> J_e_p = J_e_pc * J_pc_p;

	J << J_e_ksi, J_e_p;
	return J;
}
Matrix3d ReprojectionError::skew(const Vector3d& phi)
{
	Matrix3d Phi;
	Phi << 0, -phi(2), phi(1),
		phi(2), 0, -phi(0),
		-phi(1), phi(0), 0;
	return Phi;
}

void LMOptimizationBA::PrepareOptimization()
{
	Hcc.resize(cameras.size(), JacobianCC::Zero());
	Hpp.resize(points.size(), JacobianPP::Zero());
	Hcp.resize(cameras.size(), vector<JacobianCP>(points.size(), JacobianCP::Zero()));

	bc.resize(6 * cameras.size());
	bp.resize(3 * points.size());

	deltacs.resize(cameras.size(), Vector6d::Zero());
	deltaps.resize(points.size(), Vector3d::Zero());
}
void LMOptimizationBA::ConstructJacobian()
{
	Hcc.resize(cameras.size(), JacobianCC::Zero());
	Hpp.resize(points.size(), JacobianPP::Zero());
	Hcp.resize(cameras.size(), vector<JacobianCP>(points.size(), JacobianCP::Zero()));
	bc.setZero();
	bp.setZero();

	for (int k = 0; k < terms.size(); k++)
	{
		ReprojectionError* term = terms[k];
		JacobianType J = term->Jacobian(cameras[term->camIdx], points[term->pointIdx]);
		Matrix<double, 2, 6> Jc = J.block(0, 0, 2, 6);
		Matrix<double, 2, 3> Jp = J.block(0, 6, 2, 3);

		JacobianCC JcTJc = Jc.transpose() * Jc;
		//Hcc[term->camIdx] += Jc.transpose() * Jc + lambda * JacobianCC::Identity();
		Hcc[term->camIdx] += JcTJc + lambda * JacobianCC(JcTJc.diagonal().asDiagonal());

		JacobianPP JpTJp = Jp.transpose() * Jp;
		//Hpp[term->pointIdx] += Jp.transpose() * Jp + lambda * JacobianPP::Identity();
		Hpp[term->pointIdx] += JpTJp + lambda * JacobianPP(JpTJp.diagonal().asDiagonal());

		Hcp[term->camIdx][term->pointIdx] += Jc.transpose() * Jp;

		Vector2d error = term->function(cameras[term->camIdx], points[term->pointIdx]);
		bc.segment(6 * term->camIdx, 6) += -Jc.transpose() * error;
		bp.segment(3 * term->pointIdx, 3) += -Jp.transpose() * error;
	}

	cout << "Jacobian Constructed." << endl;
}
void LMOptimizationBA::ComputeUpdate()
{
	int Nc = cameras.size();
	int Np = points.size();

	MatrixXd A(6 * Nc, 6 * Nc);
	A.setZero();
	VectorXd b(6 * Nc);
	b.setZero();
	vector<Matrix3d> InvHpp(Np);

	// Inverse Hpp
#pragma omp parallel for
	for (int k = 0; k < Np; k++)
	{
		InvHpp[k] = Hpp[k].inverse();
	}
	cout << "Inverse Hpp Finished." << endl;

	// Construct Hcc-Hcp*invHpp*Hpc
#pragma omp parallel for
	for (int k = 0; k < Nc; k++)
	{
		A.block(6 * k, 6 * k, 6, 6) = Hcc[k];
	}
	//Hcp*invHpp
	MatrixXd HcpInvHpp(6 * Nc, 3 * Np);
	HcpInvHpp.setZero();
#pragma omp parallel for
	for (int i = 0; i < Nc; i++)
	{
#pragma omp parallel for
		for (int j = 0; j < Np; j++)
		{
			HcpInvHpp.block(6 * i, 3 * j, 6, 3) = Hcp[i][j] * InvHpp[j];
		}
	}
	// Hcc-Hcp*invHpp*Hpc
	MatrixXd Hpc(3 * Np, 6 * Nc);
	Hpc.setZero();
#pragma omp parallel for
	for (int i = 0; i < Nc; i++)
	{
#pragma omp parallel for
		for (int j = 0; j < Np; j++)
		{
			Hpc.block(3 * j, 6 * i, 3, 6) = Hcp[i][j].transpose();
		}
	}
	A += -HcpInvHpp * Hpc;
	cout << "Calculate Hcc-Hcp*invHpp*Hpc finished." << endl;

	//bc-Hcp*invHpp*bp
	b = bc - HcpInvHpp * bp;

	cout << "Solving Increment Equation..." << endl;
	VectorXd deltac = A.colPivHouseholderQr().solve(b);
	cout << "Solve Increment Equation Finished." << endl;

#pragma omp parallel for
	for (int i = 0; i < Nc; i++)
	{
		deltacs[i] = deltac.segment(6 * i, 6);
	}
	//Hpp*deltap=bp-Hpc*deltac
	VectorXd Hppdeltap = bp - Hpc * deltac;
#pragma omp parallel for
	for (int j = 0; j < Np; j++)
	{
		deltaps[j] = InvHpp[j] * Hppdeltap.segment(3 * j, 3);
	}
	cout << "Calculate Update Finished." << endl;
}
void LMOptimizationBA::ComputeNewParams()
{
	camerasUpdated.resize(cameras.size());
	pointsUpdated.resize(points.size());

	//update camera pose
#pragma omp parallel for
	for (int k = 0; k < cameras.size(); k++)
	{
		SE3Quat update = SE3Quat::exp(deltacs[k]);
		camerasUpdated[k] = update * cameras[k];
	}
	//update point
#pragma omp parallel for
	for (int k = 0; k < points.size(); k++)
	{
		pointsUpdated[k] = points[k] + deltaps[k];
	}
	cout << "Calculate New Params Finished." << endl;
}
double LMOptimizationBA::TotalError(const vector<SE3Quat>& cameras, const vector<Vector3d>& points) const
{
	double sum = 0;
	for (int k = 0; k < terms.size(); k++)
	{
		sum += terms[k]->function(cameras[terms[k]->camIdx], points[terms[k]->pointIdx]).squaredNorm();
	}
	return 0.5 * sum;
}
void LMOptimizationBA::Optimize(int niters)
{
	cout << "Optimize with " << cameras.size() << " poses, " << points.size() << " points, and " << terms.size() << " observations." << endl;
	PrepareOptimization();

	lambda = 1e-1;
	for (int k = 0; k < niters; k++)
	{
		ConstructJacobian();
		ComputeUpdate();
		ComputeNewParams();

		if (TotalError(camerasUpdated, pointsUpdated) < TotalError(cameras, points))
		{
			cameras = camerasUpdated;
			points = pointsUpdated;
			lambda /= 10;
		}
		else
		{
			lambda *= 10;
		}
		cout << "iteration = " << k << " , error = " << TotalError(cameras, points) << " , lambda = " << lambda << endl;
	}
}

void LMOptimizationBA::SaveKeyFrameTrajectoryToFile(const char* file, vector<double> TimeStamps)
{
	ofstream f(file);
	if (!f)
	{
		cerr << "Open File " << file << " failed." << endl;
		return;
	}
	f << fixed;

	for (int k = 0; k < cameras.size(); k++)
	{
		SE3Quat pose = cameras[k];
		Quaterniond  q = pose.rotation();
		Vector3d t = pose.translation();
		f << setprecision(6) << TimeStamps[k] << setprecision(7) << " " << t[0] << " " << t[1] << " " << t[2] << " ";
		f << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << endl;
	}

	f.close();
}
#define _CRT_SECURE_NO_WARNINGS
#include"Dataset.h"
#include"LMOptimizationBA.h"

int main(int argc,char** argv)
{
	if (argc != 2)
	{
		cerr << "usage: testba data.txt" << endl;
		return -1;
	}
	Dataset data;
	data.LoadFile(argv[1]);
	vector<double> TimeStamps;
	cout << "cameras: " << data.NumCameras << ", points: " << data.NumPoints << ", observations: " << data.NumObservations << endl;
	
	LMOptimizationBA optimizer;
	optimizer.InitializeStorage(data.NumCameras, data.NumPoints, data.NumObservations);
	for (map<int,Dataset::Camera>::iterator it=data.Cameras.begin(),iend=data.Cameras.end();it!=iend;it++)
	{
		Quaterniond q(it->second.q);
		Vector3d t = it->second.t;
		optimizer.addCamera(SE3Quat(q,t));
		TimeStamps.push_back(it->second.TimeStamp);
	}
	for (map<int,Dataset::Point>::iterator it=data.Points.begin(),iend=data.Points.end();it!=iend;it++)
	{
		Vector3d p = it->second.p;
		optimizer.addPoint(p);
	}

	struct CamParam
	{
		double fx, fy, cx, cy;
	};
	CamParam param = { 707.0912 ,707.0912 ,601.8873 ,183.1104 };
	for (int k = 0; k < data.NumObservations; k++)
	{
		Dataset::Observation obs = data.Observations[k];

		ReprojectionError* e = new ReprojectionError();
		e->setCameraParam(param.fx,param.fy,param.cx,param.cy);
		e->setMeasurement(obs.u,obs.v);
		e->camIdx = data.CamIndex[obs.CamIdx];
		e->pointIdx = data.PointIndex[obs.PointIdx];
		optimizer.addTerm(e);
	}
	optimizer.SaveKeyFrameTrajectoryToFile("before.txt",TimeStamps);
	cout << "Totally " << optimizer.terms.size() << " terms." << endl;
	cout << "start Optimization." << endl;
	optimizer.Optimize(10);
	optimizer.SaveKeyFrameTrajectoryToFile("after.txt", TimeStamps);
	return 0;
}
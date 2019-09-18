#pragma once
#include<iostream>
#include<fstream>
#include<map>
#include<vector>
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Geometry>
using namespace std;
using namespace Eigen;
class Dataset
{
public:
	bool LoadFile(char* file)
	{
		ifstream f(file);
		if (!f)
		{
			cerr << "Open File Failed." << endl;
			return false;
		}
		f >> NumCameras >> NumPoints >> NumObservations;

		for (int k = 0; k < NumObservations; k++)
		{
			Observation obs;
			f >> obs.CamIdx >> obs.PointIdx >> obs.u >> obs.v;
			Observations.push_back(obs);
		}

		for (int k = 0; k < NumCameras; k++)
		{
			Camera cam;
			f >> cam.Id >> cam.TimeStamp;
			f >> cam.t[0] >> cam.t[1] >> cam.t[2];
			f >> cam.q[0] >> cam.q[1] >> cam.q[2] >> cam.q[3];
			Cameras[cam.Id] = cam;
			CamIndex[cam.Id] = k;
		}

		for (int k = 0; k < NumPoints; k++)
		{
			Point p;
			f >> p.Id >> p.p[0] >> p.p[1] >> p.p[2];
			Points[p.Id] = p;
			PointIndex[p.Id] = k;
		}

		cout << "Load File Successful." << endl;
		return true;
	}
	struct Camera
	{
		int Id;
		double TimeStamp;
		Vector4d q;
		Vector3d t;
	};
	struct Point
	{
		int Id;
		Vector3d p;
	};
	struct Observation
	{
		int CamIdx, PointIdx;
		double u, v;
	};

	int NumCameras;
	int NumPoints;
	int NumObservations;

	map<int, int> CamIndex;
	map<int, int>PointIndex;
	map<int, Camera> Cameras;
	map<int, Point> Points;
	vector<Observation> Observations;
};
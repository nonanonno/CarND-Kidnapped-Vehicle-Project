/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles;i++){
		Particle p = {
				i,
				dist_x(gen),
				dist_y(gen),
				dist_theta(gen),
				1,
				{},
				{},
				{}
		};
		particles.push_back(std::move(p));
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for(auto &p : particles){
		yaw_rate = yaw_rate + dist_theta(gen);
		if(yaw_rate == 0){
			p.x += velocity * std::sin(p.theta) * delta_t + dist_x(gen);
			p.y += velocity * std::cos(p.theta) * delta_t + dist_y(gen);
		}else{
			p.x += (velocity / yaw_rate) * (std::sin(p.theta + yaw_rate * delta_t) - std::sin(p.theta));
			p.y += (velocity / yaw_rate) * (std::cos(p.theta) - std::cos(p.theta + yaw_rate * delta_t));
			p.theta += yaw_rate * delta_t;
		}
	}
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double weight_sum = 0;
	for (auto &p : particles)
	{
		p.associations = {};
		p.sense_x = {};
		p.sense_y = {};
		double weight = 1;
		for (auto ob : observations)
		{
			auto map_obs = p.toMapCoordinate(ob);
			double closest_distance = INFINITY;
			LandmarkObs closest{};
			for (const auto &lm : map_landmarks.landmark_list)
			{
				auto distance = dist(lm.x_f, lm.y_f, map_obs.x, map_obs.y);
				if(distance < closest_distance){
					closest.x = lm.x_f;
					closest.y = lm.y_f;
					closest.id = lm.id_i;
					closest_distance = distance;
				}
			}
			auto w = multiv_prob(std_landmark[0], std_landmark[1], map_obs.x, map_obs.y, closest.x, closest.y);
			p.associations.push_back(closest.id);
			p.sense_x.push_back(map_obs.x);
			p.sense_y.push_back(map_obs.y);
			
			weight *= w;
		}
		p.weight = weight;
		weight_sum += weight;
	}
	for(auto &p : particles){
		p.weight /= weight_sum;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	std::vector<double> weight;
	for(auto &p : particles){
		weight.push_back(p.weight);
	}
	std::discrete_distribution<size_t> dist_index(weight.begin(), weight.end());
	std::vector<Particle> p2;
	for (int i = 0; i < num_particles;i++){
		p2.push_back(particles[dist_index(gen)]);
	}
	particles = p2;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

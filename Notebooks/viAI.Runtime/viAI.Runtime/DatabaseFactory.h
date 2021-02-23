#pragma once

#include "NetInput.h"
#include <string>
#include <pqxx/connection.hxx>
#include <pqxx/pqxx>
#include <iostream> 
#include <assert.h>
#include <vector>

class DatabaseFactory
{
public:
	DatabaseFactory(const std::string& host, const std::string &database, 
		const std::string &user, const std::string &passwd, const std::string& port);
	DatabaseFactory(const std::string& connectionString);
	std::vector<NetInput> CreateSelectFramesByCategoryCursor(const std::string& category, const int& limit);
	~DatabaseFactory();

private:
	pqxx::connection _connection;
	std::string _connectionString;
	void CreateConnectionString();

	std::string _host;
	std::string _database;
	std::string _user;
	std::string _password;
	std::string _port;

};

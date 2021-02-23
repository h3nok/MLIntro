#include <typeinfo>
#include "DatabaseFactory.h"
#include "NetInput.h"
#include <string>
#include <samples/slog.hpp>

DatabaseFactory::DatabaseFactory(const std::string & host, const std::string & database,
	const std::string & user, const std::string & passwd, const std::string & port)
{
	this->_host = host;
	this->_database = database;
	this->_user = user;
	this->_password = passwd;
	this->_port = port;
}

DatabaseFactory::DatabaseFactory(const std::string & connectionString): _connection(connectionString)
{
	this->_connectionString = connectionString;
}

std::vector<NetInput> DatabaseFactory::CreateSelectFramesByCategoryCursor(const std::string & category, const int & limit)
{
	try {
		pqxx::connection connection(this->_connectionString);
		pqxx::work transaction(connection);
		pqxx::result result{ transaction.exec_params("SELECT frame_id, category, frame_data FROM vinet.get_frames( $1, $2)", category, limit) };

		assert(!result.empty());
		std::vector<NetInput> vec;
		
		for (auto row : result)
		{
			vec.push_back(NetInput(row));
		}
		
		return vec;
	}
	catch (const std::exception& error) {
		slog::err << error.what() << slog::endl;
		throw error;
	}
}

DatabaseFactory::~DatabaseFactory()
{
}

void DatabaseFactory::CreateConnectionString()
{

}



#pragma once
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;


bool file_exists(const std::string &path, fs::file_status s = fs::file_status{})
{
	const fs::path p = fs::path(path);

	if (fs::status_known(s) ? fs::exists(s) : fs::exists(p))
		return true;

	return false;
}

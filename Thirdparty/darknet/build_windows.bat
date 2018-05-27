md build
cd build
cmake .. -G "Visual Studio 14 2015 Win64"
cmake --build . --config Release --target ALL_BUILD
pause
#include <iostream>
using namespace std;
int main(){
#if defined(SYCL_LANGUAGE_VERSION)
cout << "SYCL_LANGUAGE_VERSION="<<SYCL_LANGUAGE_VERSION<<endl;
#endif
return 0;}

#ifndef LPBOCL_H
#define LPBOCL_H

#include <stdlib.h>
#include "CL/cl.h"
#include "LogPolar.h"


class LPBocl : public LogPolar{
  protected:	
    cl_mem xc_d, yc_d;
    cl_mem e_d, n_d;

    /* Host/device data structures */
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_int i, clerr;

    /* Program/kernel data structures */
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;


  public:
    LPBocl(Image *i, bool inv);
    ~LPBocl();
    void create_map();
    void to_cortical();
    void to_cartesian();
    void process();
};

#endif

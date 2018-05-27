include CommonDefs.mak

BIN_DIR = Bin

INC_DIRS = \
	../../Include \
	../MWClosestPoint \
	../Common

SRC_FILES = *.cpp

USED_LIBS += OpenNI2 MWClosestPoint

EXE_NAME = MWClosestPointApp

CFLAGS += -Wall


ifndef OPENNI2_INCLUDE
    $(error OPENNI2_INCLUDE is not defined. Please define it or 'source' the OpenNIDevEnvironment file from the installation)
else ifndef OPENNI2_REDIST
    $(error OPENNI2_REDIST is not defined. Please define it or 'source' the OpenNIDevEnvironment file from the installation)
endif

INC_DIRS += $(OPENNI2_INCLUDE)

include CommonCppMakefile

.PHONY: copy-redist
copy-redist:
	cp -R $(OPENNI2_REDIST)/* $(OUT_DIR)

$(OUTPUT_FILE): copy-redist


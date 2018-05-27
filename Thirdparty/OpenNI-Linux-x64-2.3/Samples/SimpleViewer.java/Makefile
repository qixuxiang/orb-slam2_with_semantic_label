include CommonDefs.mak

BIN_DIR = Bin

SRC_FILES = \
	src/org/openni/Samples/SimpleViewer/*.java

JAR_NAME = org.openni.Samples.SimpleViewer
USED_JARS = org.openni
MAIN_CLASS = org.openni.Samples.SimpleViewer.SimpleViewerApplication


ifndef OPENNI2_INCLUDE
    $(error OPENNI2_INCLUDE is not defined. Please define it or 'source' the OpenNIDevEnvironment file from the installation)
else ifndef OPENNI2_REDIST
    $(error OPENNI2_REDIST is not defined. Please define it or 'source' the OpenNIDevEnvironment file from the installation)
endif

INC_DIRS += $(OPENNI2_INCLUDE)

include CommonJavaMakefile

.PHONY: copy-redist
copy-redist:
	cp -R $(OPENNI2_REDIST)/* $(OUT_DIR)

$(OUTPUT_FILE): copy-redist


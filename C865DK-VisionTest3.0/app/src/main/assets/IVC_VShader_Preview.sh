attribute vec4 position;
attribute vec2 inputTextureCoordinate;
varying vec2 vTexCoord;
uniform mat4 mvpMatrix;
void main(){
	gl_Position = mvpMatrix*position;
	vTexCoord = inputTextureCoordinate;
}
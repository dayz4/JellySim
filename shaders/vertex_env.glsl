layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aOffset;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 offset;

out vec3 ourColor;

void main()
{
    gl_Position = projection * view * (model * vec4(aPos.x, aPos.y, aPos.z, 1.0) + vec4(aOffset.x, aOffset.y, aOffset.z, 1.0));
    ourColor = vec3(1.0f, 0.75f, 1.0f);
}
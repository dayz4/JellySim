layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aOffset;
layout (location = 2) in float aActivated;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 ourColor;

void main()
{
    gl_Position = projection * view * model * vec4(aPos.x + aOffset.x, aPos.y + aOffset.y, aPos.z + aOffset.z, 1.0);
    if (aActivated == 1.0f) {
        ourColor = vec3(0.0f, 0.0f, 1.0f);
    } else if (aActivated > 0.0f) {
        ourColor = vec3(0.0f, 0.8f, 1.0f);
    } else {
        ourColor = vec3(1.0f, 0.5f, 0.2f);
    }
}
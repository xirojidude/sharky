
Shader "Skybox/Abstract Orchard"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _SunDir ("Sun Dir", Vector) = (-.11,.07,0.99,0) 
        _XYZPos ("XYZ Offset", Vector) = (0, 15, -.25 ,0) 
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            uniform sampler2D _MainTex; 
            float4 _SunDir,_XYZPos;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 worldPos : TEXCOORD1;
                float4 vertex : SV_POSITION;   //pos
                float4 screenPos : TEXCOORD2;
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                o.worldPos = mul(unity_ObjectToWorld, v2.vertex);
                o.screenPos  = ComputeScreenPos(o.vertex);
                //o.screenPos.z = -(mul(UNITY_MATRIX_MV, vertex).z * _ProjectionParams.w);     // old calculation, as I used the depth buffer comparision for min max ray march. 

                return o;
            }



         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin



                return fragColor;
            }

            ENDCG
        }
    }
}


// Abstract Orchard
// by Dave Hoskins. October 2019
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

int spointer;
vec3 sunLight;
#define SUN_COLOUR vec3(1., .9, .8)
#define FOG_COLOUR vec3(1., .7, .7)

struct Stack
{
    vec3 pos;
    float alpha;
    float dist;
    int mat;

};

#define STACK_SIZE 8
Stack stack[STACK_SIZE];

//==============================================================================
//--------------------------------------------------------------------------
float getGroundHeight(vec2 p)
{
    float y =(sin(p.y*.23)+cos(p.x*.18))*.8;
    return y;
}
//--------------------------------------------------------------------------
mat3 getCamMat( in vec3 ro, in vec3 ta, float cr )
{
    vec3 cw = normalize(ta-ro);
    vec3 cp = vec3(sin(cr), cos(cr),0.0);
    vec3 cu = normalize( cross(cw,cp) );
    vec3 cv = normalize( cross(cu,cw) );
    return mat3( cu, cv, cw );
}

//--------------------------------------------------------------------------
// Loop the camposition around a uneven sine and cosine, and default the time 0
// to be steep at a loop point by adding 140...
vec3 getCamPos(float t)
{
    //t = sin(t*.01)*200.;
    t+=140.;
    vec3 p = vec3(3.0+50.0*sin(t*.03),
                  1.5,
                  4.0 + 50.0*cos(t*.044));
    p.y-=getGroundHeight(p.xz);
    return p;
}

//----------------------------------------------------------------------------------------
//  1 out, 1 in...
float hash11(float p)
{
    p = fract(p * .1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}
//----------------------------------------------------------------------------------------
//  1 out, 2 in...
float hash12(vec2 p)
{
    vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

//  1 out, 3 in...
vec3 hash31(float p)
{
   vec3 p3 = fract(vec3(p) * vec3(.1031, .1030, .0973));
   p3 += dot(p3, p3.yzx+33.33);
   return fract((p3.xxy+p3.yzz)*p3.zyx); 
}

//  3 out, 3 in...
vec3 hash33(vec3 p3)
{
    p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+33.33);
    return fract((p3.xxy + p3.yxx)*p3.zyx);
}


//------------------------------------------------------------------------------
float randomTint(vec3 pos)
{
    float r = texture(iChannel1, pos.xz*.0027).x;
    return r+.5;
}

//----------------------------------------------------------------------------------------
vec3 texCube(sampler2D sam, in vec3 p, in vec3 n )
{
    vec3 x = texture(sam, p.yz).xyz;
    vec3 y = texture(sam, p.zx).xyz;
    vec3 z = texture(sam, p.xy).xyz;
    return (x*abs(n.x) + y*abs(n.y) + z*abs(n.z))/(abs(n.x)+abs(n.y)+abs(n.z));
}

//------------------------------------------------------------------------------
vec4 grassTexture(vec3 pos, vec3 nor)
{
    
    float g = texture(iChannel1, pos.xz*.5).x;
    float s = texture(iChannel1, pos.xz*.015).x*.2;
    
    
    vec3 flower = texture(iChannel2, pos.xz*.15).xyz;
    float rand = texture(iChannel1, pos.xz*.003).x;
    rand *= rand*rand;
    
    flower =pow(flower,vec3(8, 15, 5)) *10. * rand;
    vec4 mat = vec4(g*.05+s, g*.65, 0, g*.1);
    mat.xyz += flower;
    
    return min(mat, 1.0);
}

//------------------------------------------------------------------------------
vec4 barkTexture(vec3 p, vec3 nor)
{
    vec2 r = floor(p.xz / 5.0) * 0.02;
    
    float br = texture(iChannel1, r).x;
    vec3 mat = texCube(iChannel3, p, nor) * vec3(.35, .25, .25);
    mat += texCube(iChannel3, p*1.73, nor)*smoothstep(0.0,.2, mat.y)*br * vec3(1,.9,.8);
    //mat*=mat*2.5;
    return vec4(mat, .1);
}

//------------------------------------------------------------------------------
vec4 leavesTexture(vec3 p, vec3 nor)
{
    
    vec3 rand = texCube(iChannel2, p*.15,nor);
    vec3 mat = vec3(0.4,1.2,0) *rand;
    if (nor.y < 0.0) mat += vec3(1., 0.5,.5);
    
    return vec4(mat, .0);
}

//------------------------------------------------------------------------------
vec4 fruitTexture(vec3 p, vec3 nor, float i)
{
    
    
    float rand = texCube(iChannel2, p*.1 ,nor).x;
    float t = dot(nor, normalize(vec3(.8, .1, .1)));
    vec3 mat = vec3(1.,abs(t)*rand,0);
    mat = mix(vec3(0,1,0), mat, i/10.);

    return vec4(mat, .5);
}



//------------------------------------------------------------------------------
float distanceRayPoint(vec3 ro, vec3 rd, vec3 p, out float h)
{
    h = dot(p-ro,rd);
    return length(p-ro-rd*h);
}

//------------------------------------------------------------------------------
const int   SEEDS = 8 ;
const float STEP_SIZE = 2.;
#define SIZE .03


// This seed code is the starfield stuff from iapafoto
// I've just removed the alpha part...
// https://www.shadertoy.com/view/Xl2BRR
mat2 rotMat2D(float a)
{
    float si = sin(a);
    float co = cos(a);
    return mat2(si, co, -co, si);
}

vec3 floatingSeeds(in vec3 ro, in vec3 rd, in float tmax)
{ 
 
    float d =  0.;
    ro /= STEP_SIZE;
    vec3 pos = floor(ro),
         ri = 1./rd,
         rs = sign(rd),
         dis = (pos-ro + .5 + rs*0.5) * ri;
    
    float dint;
    vec3 offset, id;
    vec3 col = vec3(0);
    vec3 sum = vec3(0);
    //float size = .04;
    
    for( int i=0; i< SEEDS; i++ )
    {
        id = hash33(pos);

        offset = clamp(id+.2*cos(id*iTime),SIZE, 1.-SIZE);
        d = distanceRayPoint(ro, rd, pos+offset, dint);
        
        if (dint > 0. && dint * STEP_SIZE < tmax)
        {
            col = vec3(.4)*smoothstep(SIZE, 0.0,d);
            sum += col;
        }
        vec3 mm = step(dis.xyz, dis.yxy) * step(dis.xyz, dis.zzx);
        dis += mm * rs * ri;
        pos += mm * rs;
    }
  
    return sum * .7;
}

//--------------------------------------------------------------------------
float findClouds2D(in vec2 p)
{
    float a = 1.5, r = 0.0;
    p*= .000001;
    for (int i = 0; i < 5; i++)
    {
        r+= texture(iChannel1,p*=2.2).x*a;
        a*=.5;
    }
    return max(r-1.5, 0.0);
}
//------------------------------------------------------------------------------
// Use the difference between two cloud densities to light clouds in the direction of the sun.
vec4 getClouds(vec3 pos, vec3 dir)
{
    if (dir.y < 0.0) return vec4(0.0);
    float d = (4000. / dir.y);
    vec2 p = pos.xz+dir.xz*d;
    float r = findClouds2D(p);
    float t = findClouds2D(p+normalize(sunLight.xz)*30.);    
    t = sqrt(max((r-t)*20., .2))*2.;
    vec3 col = vec3(t) * SUN_COLOUR;
    // returns colour and alpha...
    return vec4(col, r);
} 


//------------------------------------------------------------------------------
// Thanks to Fizzer for the space-folded tree idea...
/// https://www.shadertoy.com/view/4tVcWR
vec2 map(vec3 p, float t)
{
 
    float matID, f;
    p.y += getGroundHeight(p.xz);
    float num = (floor(p.z/5.))*5.+(floor(p.x/5.0))*19.;
    p.xz = mod(p.xz, 5.0)-2.5;
    //p.xz *= rotMat2D(p.y*num/300.); // ... No, just too expensive. :)
    
    float d = p.y;
    matID = 0.0;

    float s=1.,ss=1.6;
    
    // Tangent vectors for the branch local coordinate system.
    vec3 w=normalize(vec3(-1.5+abs(hash11(num*4.)*.8),1,-1.));
    vec3 u=normalize(cross(w,vec3(0,1.,0.)));

    float scale=3.5;
    p/=scale;
    vec3 q = p;
    // Make the iterations lessen over distance for speed up...
    int it = 10-int(min(t*.03, 9.0));

    float h  = hash11(num*7.)*.3+.3;
    vec3 uwc = normalize(cross(u,w));
    int dontFold = int(hash11(num*23.0) * 9.0)+3;
    
    float thick = .2/(h-.24);
    for (int i = 0; i < it; i++)
    {
        f = scale*max(p.y-h,max(-p.y,length(p.xz)-.06/(p.y+thick)))/s;
        if (f <= d)
        {
            d = f;
            matID = 1.0;
        }

        // Randomly don't fold the space to give more branch types...
        if (i != dontFold)
            p.xz = abs(p.xz);

        p.y-=h;
        p*=mat3(u,uwc,w);
        p*=ss;
        s*=ss;
    }

    float fr = .2;
    f = (length(p)-fr)/s;
    if (f <= d)
    {
        d = f;
        matID = 2.0;
    }
    
    q.y -= h*1.84;
    h *= 1.1;
    for (int i = 0; i < it; i++)
    {
        p = (normalize(hash31(num+float(i+19))-.5))*vec3(h, 0.1, h);
        p+=q;
        float ds =length(p)-.015;
        if (ds <= d)
        {
            matID = 3.0+float(i);
            d = ds;
        }
    }

    return vec2(d, matID);
}

//------------------------------------------------------------------------------
float sphereRadius(float t)
{
    t = abs(t-.0);
    t *= 0.003;
    return clamp(t, 1.0/iResolution.y, 3000.0/iResolution.y);
}

//------------------------------------------------------------------------------
float shadow( in vec3 ro, in vec3 rd, float dis)
{
    float res = 1.0;
    float t = hash11(dis)*.5+.2;
    float h;
    
    for (int i = 0; i < 10; i++)
    {
        vec3 p =  ro + rd*t;

        h = map(p,dis).x;
        res = min(10.*h / t*t, res);
        t += h*2.5;
    }
    //res += t*t*.02; // Dim over distance
    return clamp(res, .3, 1.0);
}

//-------------------------------------------------------------------------------------------
// Taken almost straight from Inigo's shaders, thanks man!
// But I've changed a few things like the for-loop is now a float,
// which removes the need for the extra multiply and divide in GL2
float calcOcc( in vec3 pos, in vec3 nor, float d )
{
    float occ = 0.0;
    float sca = 1.0;
    for(float h= 0.07; h < .25; h+= .07)
    {
        vec3 opos = pos + h*nor;
        float d = map( opos, d ).x;
        occ += (h-d)*sca;
        sca *= 0.95;
    }
    return clamp( 1.0 - 2.0*occ, 0.0, 1.0 );
}

//-------------------------------------------------------------------------------------------
float marchScene(in vec3 rO, in vec3 rD, vec2 co)
{
    float t = hash12(co)*.5;
    vec4 normal = vec4(0.0);
    vec3 p;
    float alphaAcc = 0.0;

    spointer = 0;
    for( int j=min(0,iFrame); j < 140; j++ )
    {
        // Check if it's full or too far...
        if (spointer == STACK_SIZE || alphaAcc >= 1.) break;
        p = rO + t*rD;
        float sphereR = sphereRadius(t);
        vec2 h = map(p, t);
        if( h.x <= sphereR)
        {
            //h = max(h,0.0);
            float alpha = (1.0 - alphaAcc) * min(((sphereR-h.x+.01) / sphereR), 1.0);
            stack[spointer].pos = p;
            stack[spointer].alpha = alpha;
            stack[spointer].dist = t;
            stack[spointer].mat = int(h.y);
            alphaAcc += alpha;
            spointer++;
        }
        t +=  h.x+t*0.007;
    }
    return alphaAcc;
}   

//-------------------------------------------------------------------------------------------
vec3 lighting(in int id, in vec4 mat, in vec3 pos, in vec3 normal, in vec3 eyeDir, in float d)
{

    // Shadow and local occlusion...
    float sh = shadow(pos+sunLight*.01,  sunLight, d);
    float occ = calcOcc(pos, normal, d);
    
    // Light surface with 'sun'...
    vec3 col = mat.xyz * SUN_COLOUR*(max(dot(sunLight,normal), 0.0))*sh;
    if (id == 2 && normal.y < -0.5)
    {
        col.y += .15;
    }


    // Ambient...
    
    float fre = clamp(1.0+dot(normal,eyeDir),0.0,1.0);
    
    float bac = clamp(dot( normal, normalize(vec3(-sunLight.x,0.0,-sunLight.z))), 0.0, 1.0 );
    normal = reflect(eyeDir, normal); // Specular...
    col += pow(max(dot(sunLight, normal), 0.0), 16.0)  * SUN_COLOUR * sh * mat.w * occ * fre;
    col += bac*mat.xyz* .3 * occ;
    col += mat.xyz * max(normal.y, 0.0) * occ * .5;
    //col += SUN_COLOUR * fre *.2*occ;

    return min(col, 1.0);
}

//------------------------------------------------------------------------------
vec3 getNormal2(vec3 p, float e)
{
    return normalize( vec3( map(p+vec3(e,0.0,0.0), 0.).x - map(p-vec3(e,0.0,0.0), 0.).x,
                            map(p+vec3(0.0,e,0.0), 0.).x - map(p-vec3(0.0,e,0.0), 0.).x,
                            map(p+vec3(0.0,0.0,e), 0.).x - map(p-vec3(0.0,0.0,e), 0.).x));
}

vec3 getNormal(vec3 pos, float ds)
{

    float c = map(pos, 0.).x;
    // Use offset samples to compute gradient / normal
    vec2 eps_zero = vec2(ds, 0.0);
    return normalize(vec3(map(pos + eps_zero.xyy, 0.0).x, map(pos + eps_zero.yxy, 0.0).x,
                          map(pos + eps_zero.yyx, 0.0).x) - c);
}


//------------------------------------------------------------------------------
vec3 getSky(vec3 dir)
{
    vec3 col = mix(vec3(FOG_COLOUR), vec3(.0, 0.2,0.4),(abs(dir.y)));
    return col;
}


//==============================================================================
void mainImage( out vec4 fragColour, in vec2 fragCoord )
{
    vec2 mouseXY = iMouse.xy / iResolution.xy;
    vec2 uv = (-iResolution.xy + 2.0 * fragCoord ) / iResolution.y;
    sunLight = normalize(vec3(-.8,1.8,-1.5));
    
    // Camera stuff...
    float time = iTime+mouseXY.x*100.0;
    vec3 camera = getCamPos(time);
    vec3 lookat = getCamPos(time+10.);
    float ride = sin(iTime*.3)+.3;
    camera.y += ride;
    lookat.y += ride;
    
    mat3 camMat = getCamMat(camera, lookat, 0.0);
    vec3 seedDir = normalize( vec3(uv, cos((length(uv*.4)))));
    vec3 rd = camMat * seedDir;

    vec3 col = vec3(0);

    vec3 sky  = getSky(rd);
  

    // Build the stack returning the final alpha value...
    float alpha = marchScene(camera, rd, fragCoord);
    vec4 mat;
    // Render the stack...
    if (alpha > .0)
    {
        for (int i = 0; i < spointer; i++)
        {
            vec3  pos = stack[i].pos; 
            float d = stack[i].dist;
            
            vec3 nor =  getNormal(pos, sphereRadius(d));
            int matID = stack[i].mat;
            if (matID == 0) mat =  grassTexture(pos, nor);
            else
                if (matID == 1) mat = barkTexture(pos, nor);
            else
            if (matID == 2)
            {
                mat = leavesTexture(pos, nor);
                
                
            }
            else
                mat = fruitTexture(pos, nor, float(matID - 3));

            mat *= randomTint(pos);
 
            vec3  temp = lighting(matID,mat, pos, nor, rd, d);
            if (matID == 3) temp=temp*.4+vec3(.15, .01,0);
            
            temp = mix(sky, temp , exp(-d*.01));
            col += temp * stack[i].alpha;
        }
    }
    vec4 cc = getClouds(camera, rd);
    sky+= pow(max(dot(sunLight, rd), 0.0), 300.0)*SUN_COLOUR*2.;
    sky = mix(sky, cc.xyz, cc.w);
    col += sky *  (1.0-alpha);
    
    float d = stack[0].dist;
    col+= floatingSeeds(camera, rd, d);
    
   
    // Sun glow effect...
    col+=pow(max(dot(sunLight, rd), 0.0), 6.0)*SUN_COLOUR*.2;
    
    // Clamp and contrast...
    col = col * vec3(1., .9,.9);
    col = clamp(col,0.,1.);
    col = col*col*(3.0-2.0*col);

    
    // The usual vignette...which manages to add more vibrancy...
    vec2 q = fragCoord / iResolution.xy;
    col *= pow(90.0*q.x*q.y*(1.0-q.x)*(1.0-q.y), 0.5);
    // A nice fade in start...
    
    
    col *= smoothstep(0.0, 5.0, time);
    fragColour = vec4(sqrt(col), 1.0);
    
}

